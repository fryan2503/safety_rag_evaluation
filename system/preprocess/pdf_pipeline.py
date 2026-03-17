"""Reusable PDF processing pipeline for splitting, cropping, and summarizing."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
import pandas as pd


@dataclass
class PDFPreprocessConfig:
    """Settings controlling how the PDF is split, cropped, and summarized."""

    input_pdf: Path
    split_dir: Path
    cropped_dir: Path
    toc_level: int = 1
    min_pages: int = 1
    crop_percent: float = 0.075
    sub_level: int = 2
    min_words_for_subsplit: int = 3000
    summary_csv_initial: Path = Path("results/csvs/pdf_word_counts.csv")
    summary_csv_updated: Path = Path("results/csvs/pdf_word_counts_after_subsplit.csv")
    summary_csv_final: Path = Path("results/csvs/pdf_word_counts_final.csv")
    half_split_threshold: int | None = None
    subchapter_dir: Path | None = None
    document_label: str = "manual"

    def __post_init__(self) -> None:
        self.input_pdf = Path(self.input_pdf)
        self.split_dir = Path(self.split_dir)
        self.cropped_dir = Path(self.cropped_dir)
        self.summary_csv_initial = Path(self.summary_csv_initial)
        self.summary_csv_updated = Path(self.summary_csv_updated)
        self.summary_csv_final = Path(self.summary_csv_final)
        if self.subchapter_dir is None:
            self.subchapter_dir = self.split_dir / f"by_subchapter_L{self.sub_level}"
        else:
            self.subchapter_dir = Path(self.subchapter_dir)
        if self.half_split_threshold is None:
            self.half_split_threshold = self.min_words_for_subsplit


class PDFPreprocessor:
    """Encapsulates the multi-stage PDF preprocessing workflow."""

    def __init__(self, config: PDFPreprocessConfig):
        self.config = config
        self.config.split_dir.mkdir(parents=True, exist_ok=True)
        self.config.cropped_dir.mkdir(parents=True, exist_ok=True)
        self.config.subchapter_dir.mkdir(parents=True, exist_ok=True)
        for csv_path in (
            self.config.summary_csv_initial,
            self.config.summary_csv_updated,
            self.config.summary_csv_final,
        ):
            csv_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> pd.DataFrame:
        """Execute the full preprocessing pipeline."""
        self.split_by_toc()
        self.crop_split_pdfs()
        summary = self.write_word_counts_to_csv(
            directory=self.config.cropped_dir,
            csv_path=self.config.summary_csv_initial,
        )
        summary = self.auto_subsplit(summary)
        summary = self.half_split_oversized(summary)
        return summary

    def split_by_toc(self) -> List[Dict[str, int | str]]:
        """Split the input PDF according to the configured TOC level."""
        doc = fitz.open(str(self.config.input_pdf))
        toc = doc.get_toc(simple=True) or []
        n_pages = len(doc)

        level_idxs = [i for i, (lvl, _, _) in enumerate(toc) if lvl == self.config.toc_level]
        splits: List[Dict[str, int | str]] = []

        for idx in level_idxs:
            lvl, title, start_1b = toc[idx]

            next_start_1b = None
            for j in range(idx + 1, len(toc)):
                lvl_j, _, next_page_1b = toc[j]
                if lvl_j <= lvl:
                    next_start_1b = next_page_1b
                    break

            end_1b = (next_start_1b - 1) if next_start_1b is not None else n_pages
            start0 = max(0, start_1b - 1)
            end0 = min(n_pages - 1, end_1b - 1)
            if end0 < start0:
                continue

            pages = end0 - start0 + 1
            if pages < self.config.min_pages:
                continue

            split_info = {
                "title": title,
                "start0": start0,
                "end0": end0,
                "start_1b": start_1b,
                "end_1b": end_1b,
            }
            splits.append(split_info)

        for k, split in enumerate(splits, start=1):
            out_name = (
                f"{k:03d}__L{self.config.toc_level}__{self._safe(split['title'])}"
                f"__pp{split['start_1b']}-{split['end_1b']}.pdf"
            )
            out_path = self.config.split_dir / out_name
            part = fitz.open()
            part.insert_pdf(doc, from_page=split["start0"], to_page=split["end0"])
            part.save(out_path, deflate=True, garbage=4)
            part.close()

        doc.close()
        print(
            f"Created {len(splits)} PDFs for {self.config.document_label} in "
            f"{self.config.split_dir.resolve()}"
        )
        return splits

    def crop_split_pdfs(self) -> None:
        """Apply cropping to every split PDF."""
        for pdf_path in sorted(self.config.split_dir.glob("*.pdf")):
            output_path = self.config.cropped_dir / pdf_path.name
            self._crop_pdf(pdf_path, output_path)
            print(f"Cropped {pdf_path.name} -> {output_path.name}")
        print(f"All PDFs cropped into {self.config.cropped_dir.resolve()}")

    def generate_word_counts(self, directory: Path) -> pd.DataFrame:
        """Compute word-count statistics for every PDF in the directory."""
        rows: List[Dict[str, float | int | str]] = []
        for pdf in sorted(directory.glob("*.pdf")):
            with fitz.open(pdf) as doc:
                page_words = [len((page.get_text("text") or "").split()) for page in doc]
                rows.append(
                    {
                        "file": pdf.name,
                        "n_pages": len(doc),
                        "total_words": sum(page_words),
                        "max_words_in_a_page": max(page_words) if page_words else 0,
                        "mean_words_per_page": (sum(page_words) / len(doc)) if len(doc) else 0.0,
                    }
                )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("total_words", ascending=False).reset_index(drop=True)

    def write_word_counts_to_csv(self, directory: Path, csv_path: Path) -> pd.DataFrame:
        summary = self.generate_word_counts(directory)
        if not summary.empty:
            summary.to_csv(csv_path, index=False)
            print(f"Saved word counts to {csv_path}")
        else:
            print(f"No PDFs found in {directory} to summarize.")
        return summary

    def auto_subsplit(self, summary: pd.DataFrame) -> pd.DataFrame:
        threshold = self.config.min_words_for_subsplit
        if summary.empty:
            return summary

        targets = summary.loc[summary["total_words"] > threshold, "file"].tolist()
        if not targets:
            print(f"No files exceeded {threshold} words. Skipping sub-splitting.")
            return summary

        manual = fitz.open(str(self.config.input_pdf))
        toc = manual.get_toc(simple=True) or []
        n_pages = len(manual)
        toc_entries = [
            {"level": lvl, "title": title, "page_1b": page}
            for (lvl, title, page) in toc
        ]
        deleted_files: List[str] = []

        for fname in targets:
            split_pdf_path = self.config.split_dir / fname
            cropped_pdf_path = self.config.cropped_dir / fname
            if not split_pdf_path.exists():
                print(f"WARNING: {fname} not found in {self.config.split_dir}")
                continue

            chap_start_1b, chap_end_1b = self._parse_page_range_from_name(fname)
            sub_starts: List[Tuple[int, Dict[str, int | str]]] = []
            for i, entry in enumerate(toc_entries):
                if (
                    entry["level"] == self.config.sub_level
                    and chap_start_1b <= entry["page_1b"] <= chap_end_1b
                ):
                    sub_starts.append((i, entry))

            if not sub_starts:
                print(
                    f"No level {self.config.sub_level} entries within pages "
                    f"{chap_start_1b}-{chap_end_1b} ({fname})."
                )
                continue

            for k, (idx, entry) in enumerate(sub_starts, start=1):
                start_1b = entry["page_1b"]
                next_start_1b = None
                for j in range(idx + 1, len(toc_entries)):
                    next_entry = toc_entries[j]
                    if next_entry["page_1b"] > chap_end_1b:
                        break
                    if (
                        chap_start_1b <= next_entry["page_1b"] <= chap_end_1b
                        and next_entry["level"] <= self.config.sub_level
                    ):
                        next_start_1b = next_entry["page_1b"]
                        break

                end_1b = (next_start_1b - 1) if next_start_1b is not None else chap_end_1b
                start0 = max(0, start_1b - 1)
                end0 = min(n_pages - 1, end_1b - 1)
                if end0 < start0:
                    continue

                base_title = entry["title"]
                out_name = (
                    f"{Path(fname).stem}__L{self.config.sub_level}__{k:02d}__{self._safe(base_title)}"
                    f"__pp{start_1b}-{end_1b}.pdf"
                )
                out_path = Path(self.config.subchapter_dir) / out_name

                part = fitz.open()
                part.insert_pdf(manual, from_page=start0, to_page=end0)
                part.save(out_path, deflate=True, garbage=4)
                part.close()

                cropped_out = self.config.cropped_dir / out_path.name
                self._crop_pdf(out_path, cropped_out)
                print(f"Created subchapter {out_path.name} -> cropped {cropped_out.name}")

            if cropped_pdf_path.exists():
                cropped_pdf_path.unlink()
                deleted_files.append(fname)
                print(f"Deleted original cropped file: {fname}")

        manual.close()
        if deleted_files:
            print(f"Deleted {len(deleted_files)} cropped PDFs after sub-splitting.")
        return self.write_word_counts_to_csv(
            directory=self.config.cropped_dir,
            csv_path=self.config.summary_csv_updated,
        )

    def half_split_oversized(self, summary: pd.DataFrame) -> pd.DataFrame:
        if summary.empty:
            return summary
        threshold = self.config.half_split_threshold or self.config.min_words_for_subsplit
        large_files = summary.loc[summary["total_words"] > threshold, "file"].tolist()
        if not large_files:
            print("No oversized PDFs remaining after sub-splitting.")
            return summary

        print(f"Half-splitting {len(large_files)} PDFs (> {threshold} words)...")
        for fname in large_files:
            pdf_path = self.config.cropped_dir / fname
            if not pdf_path.exists():
                continue

            with fitz.open(pdf_path) as doc:
                n_pages = len(doc)
                if n_pages < 2:
                    continue
                mid_page = n_pages // 2 - 1

                part1_path = self.config.cropped_dir / f"{pdf_path.stem}_part01.pdf"
                part1 = fitz.open()
                part1.insert_pdf(doc, from_page=0, to_page=mid_page)
                part1.save(part1_path, deflate=True, garbage=4)
                part1.close()

                part2_path = self.config.cropped_dir / f"{pdf_path.stem}_part02.pdf"
                part2 = fitz.open()
                part2.insert_pdf(doc, from_page=mid_page + 1, to_page=n_pages - 1)
                part2.save(part2_path, deflate=True, garbage=4)
                part2.close()

            pdf_path.unlink(missing_ok=True)
            print(f"Split {fname} -> {part1_path.name}, {part2_path.name}")

        return self.write_word_counts_to_csv(
            directory=self.config.cropped_dir,
            csv_path=self.config.summary_csv_final,
        )

    @staticmethod
    def crop_pdf(
        source: Path,
        dest: Path,
        crop_percent: float = 0.075,
        crop_top: float | None = None,
        crop_bottom: float | None = None,
        crop_left: float | None = None,
        crop_right: float | None = None,
    ) -> None:
        """Crop margins from all pages of a PDF and save to *dest*.

        Applies a uniform margin crop controlled by *crop_percent* (fraction
        of page width for left/right, fraction of page height for top/bottom).
        Individual sides can be overridden with the ``crop_top``,
        ``crop_bottom``, ``crop_left``, and ``crop_right`` parameters; when
        any of these is not ``None`` it takes precedence over *crop_percent*
        for that side.

        Args:
            source: Path to the input PDF file.
            dest: Path where the cropped PDF will be saved.
            crop_percent: Default fraction applied to all four sides
                (0.075 = 7.5 %).  Ignored for any side that has an
                explicit override.
            crop_top: Fraction of page height to remove from the top.
            crop_bottom: Fraction of page height to remove from the bottom.
            crop_left: Fraction of page width to remove from the left.
            crop_right: Fraction of page width to remove from the right.

        Example::

            # Uniform 7.5 % crop on all sides
            PDFPreprocessor.crop_pdf(src, dst)

            # 7.5 % top/bottom, no side crop
            PDFPreprocessor.crop_pdf(src, dst, crop_top=0.075,
                                     crop_bottom=0.075,
                                     crop_left=0.0, crop_right=0.0)
        """
        with fitz.open(source) as doc:
            for page in doc:
                rect = page.rect
                lm = rect.width * (crop_left if crop_left is not None else crop_percent)
                rm = rect.width * (crop_right if crop_right is not None else crop_percent)
                tm = rect.height * (crop_top if crop_top is not None else crop_percent)
                bm = rect.height * (crop_bottom if crop_bottom is not None else crop_percent)
                new_rect = fitz.Rect(rect.x0 + lm, rect.y0 + tm, rect.x1 - rm, rect.y1 - bm)
                page.set_cropbox(new_rect)
            doc.save(dest, deflate=True, garbage=4)

    def _crop_pdf(self, source: Path, dest: Path) -> None:
        self.crop_pdf(source, dest, self.config.crop_percent)

    @staticmethod
    def _safe(name: str, max_len: int = 80) -> str:
        name = re.sub(r"\s+", " ", name).strip()
        name = re.sub(r"[^A-Za-z0-9 _\-\.\(\)]", "", name)
        name = name[:max_len].strip().replace(" ", "_")
        return name or "untitled"

    @staticmethod
    def _parse_page_range_from_name(name: str) -> Tuple[int, int]:
        m = re.search(r"__pp(\d+)-(\d+)\.pdf$", name)
        if not m:
            raise ValueError(f"Could not parse page range from filename: {name}")
        start_1b = int(m.group(1))
        end_1b = int(m.group(2))
        if end_1b < start_1b:
            raise ValueError(f"Invalid page range in filename: {name}")
        return start_1b, end_1b

__all__ = ["PDFPreprocessConfig", "PDFPreprocessor"]
