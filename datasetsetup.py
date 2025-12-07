import json
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd
from datasets import load_dataset


def load_dataframe_from_csv(csv_path: str, sample_size: int | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if sample_size is not None and sample_size > 0:
        df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
    return df


def load_dataframe_from_huggingface(dataset_name: str, split: str, sample_size: int | None) -> pd.DataFrame:
    dataset = load_dataset(dataset_name, split=split)
    df = dataset.to_pandas()
    if sample_size is not None and sample_size > 0:
        df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
    return df


def build_prompt_from_row(row: pd.Series, column_mapping: dict[str, str], template: str) -> str:
    values = {}
    for logical_name, column_name in column_mapping.items():
        if not column_name:
            values[logical_name] = ""
            continue
        try:
            values[logical_name] = row[column_name]
        except Exception:
            values[logical_name] = ""
    try:
        return template.format(**values)
    except Exception:
        # Fallback simple join if template has issues
        return " ".join(str(values.get(k, "")) for k in [
            "question", "solution", "difficulty", "topic"
        ] if values.get(k, "") != "")


def dataframe_to_text_rows(df: pd.DataFrame, column_mapping: dict[str, str], template: str) -> list[dict]:
    records: list[dict] = []
    for _, row in df.iterrows():
        prompt_text = build_prompt_from_row(row, column_mapping, template)
        records.append({"text": prompt_text})
    random.shuffle(records)
    return records


def split_records(records: list[dict], train_ratio: float, test_ratio: float) -> tuple[list[dict], list[dict], list[dict]]:
    total = len(records)
    train_end = int(total * train_ratio)
    test_end = train_end + int(total * test_ratio)
    train = records[:train_end]
    test = records[train_end:test_end]
    valid = records[test_end:]
    return train, test, valid


def save_jsonl(records: list[dict], out_path: str) -> None:
    with open(out_path, 'w') as f:
        for entry in records:
            f.write(json.dumps(entry) + '\n')


class DatasetUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MCES10 Software AI Dataset Utility")

        # State variables
        self.dataset_type = tk.StringVar(value="Hugging Face")
        self.csv_path = tk.StringVar(value="")
        self.hf_name = tk.StringVar(value="MCES10-Software/Python-Code-Solutions")
        self.hf_split = tk.StringVar(value="train")
        self.sample_size = tk.StringVar(value="1000")
        self.train_ratio = tk.StringVar(value="0.6667")
        self.test_ratio = tk.StringVar(value="0.1667")
        self.output_dir = tk.StringVar(value=".")

        # Dynamic column mappings: list of (placeholder, column) rows
        self.mapping_rows: list[dict] = []

        # Prompt template
        self.prompt_template = tk.StringVar(value=(
            "You are a senior software engineer who specialised in python and will answer the user's Question:{question}. "
            "Provide an acurate Solution: {solution} using the correct Topic {topic}. "
            "Then find the Difficulty: {difficulty} of the problem."
        ))

        self._build_ui()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Dataset source section
        src_frame = ttk.LabelFrame(container, text="Dataset Source", padding=8)
        src_frame.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        for i in range(6):
            src_frame.columnconfigure(i, weight=1)

        ttk.Label(src_frame, text="Type").grid(row=0, column=0, sticky="w")
        ttk.OptionMenu(src_frame, self.dataset_type, self.dataset_type.get(), "CSV", "Hugging Face", command=self._on_type_change).grid(row=0, column=1, sticky="ew")

        # CSV controls
        self.csv_label = ttk.Label(src_frame, text="CSV Path")
        self.csv_entry = ttk.Entry(src_frame, textvariable=self.csv_path)
        self.csv_btn = ttk.Button(src_frame, text="Browse", command=self._browse_csv)

        # HF controls
        self.hf_label = ttk.Label(src_frame, text="HF Dataset")
        self.hf_entry = ttk.Entry(src_frame, textvariable=self.hf_name)
        self.split_label = ttk.Label(src_frame, text="Split")
        self.split_entry = ttk.Entry(src_frame, textvariable=self.hf_split)

        # Shared controls
        ttk.Label(src_frame, text="Sample Size").grid(row=1, column=0, sticky="w")
        ttk.Entry(src_frame, textvariable=self.sample_size).grid(row=1, column=1, sticky="ew")
        ttk.Label(src_frame, text="Train Ratio").grid(row=1, column=2, sticky="w")
        ttk.Entry(src_frame, textvariable=self.train_ratio).grid(row=1, column=3, sticky="ew")
        ttk.Label(src_frame, text="Test Ratio").grid(row=1, column=4, sticky="w")
        ttk.Entry(src_frame, textvariable=self.test_ratio).grid(row=1, column=5, sticky="ew")

        ttk.Label(src_frame, text="Output Dir").grid(row=2, column=0, sticky="w")
        ttk.Entry(src_frame, textvariable=self.output_dir).grid(row=2, column=1, sticky="ew")
        ttk.Button(src_frame, text="Pick", command=self._browse_dir).grid(row=2, column=2, sticky="ew")

        # Column mapping section (dynamic)
        self.map_frame = ttk.LabelFrame(container, text="Column Mapping (add rows: placeholder ➜ dataframe column)", padding=8)
        self.map_frame.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        self.map_frame.columnconfigure(2, weight=1)

        header_idx = ttk.Label(self.map_frame, text="#")
        header_placeholder = ttk.Label(self.map_frame, text="Placeholder")
        header_column = ttk.Label(self.map_frame, text="DataFrame Column")
        header_actions = ttk.Label(self.map_frame, text="Actions")
        header_idx.grid(row=0, column=0, padx=2, sticky="w")
        header_placeholder.grid(row=0, column=1, padx=2, sticky="w")
        header_column.grid(row=0, column=2, padx=2, sticky="w")
        header_actions.grid(row=0, column=3, padx=2, sticky="w")

        # Container to hold dynamic rows to avoid grid bounds issues
        self.map_rows_container = ttk.Frame(self.map_frame)
        self.map_rows_container.grid(row=1, column=0, columnspan=4, sticky="ew")
        self.map_rows_container.columnconfigure(2, weight=1)

        # Buttons placed after the rows container (fixed position)
        btns_frame = ttk.Frame(self.map_frame)
        btns_frame.grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Button(btns_frame, text="Add Mapping", command=self._add_mapping_row).grid(row=0, column=0)
        ttk.Button(btns_frame, text="Clear All", command=self._clear_all_mappings).grid(row=0, column=1, padx=6)

        # Prompt template section
        prompt_frame = ttk.LabelFrame(container, text="Prompt Template (use placeholders you define above)", padding=8)
        prompt_frame.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)
        self.prompt_text = tk.Text(prompt_frame, height=6, wrap="word")
        self.prompt_text.insert("1.0", self.prompt_template.get())
        self.prompt_text.grid(row=0, column=0, sticky="nsew")
        prompt_frame.rowconfigure(0, weight=1)
        prompt_frame.columnconfigure(0, weight=1)

        # Actions
        action_frame = ttk.Frame(container, padding=4)
        action_frame.grid(row=3, column=0, sticky="ew")
        #ttk.Button(action_frame, text="Preview 3", command=self._preview).grid(row=0, column=0, padx=4)
        ttk.Button(action_frame, text="Run and Create Files", command=self._run).grid(row=0, column=1, padx=4)

        self._on_type_change(self.dataset_type.get())
        # Seed default mappings
        for placeholder, col in [
            ("question", "question"),
            ("solution", "solution"),
            ("difficulty", "difficulty"),
            ("topic", "topic"),
        ]:
            self._add_mapping_row(placeholder, col)

    def _on_type_change(self, selection: str) -> None:
        # Toggle visibility of CSV vs HF controls
        # Clear current placements
        for widget in [self.csv_label, self.csv_entry, self.csv_btn, self.hf_label, self.hf_entry, self.split_label, self.split_entry]:
            widget.grid_forget()

        if self.dataset_type.get() == "CSV":
            self.csv_label.grid(row=0, column=2, sticky="w")
            self.csv_entry.grid(row=0, column=3, sticky="ew")
            self.csv_btn.grid(row=0, column=4, sticky="ew")
        else:
            self.hf_label.grid(row=0, column=2, sticky="w")
            self.hf_entry.grid(row=0, column=3, sticky="ew")
            self.split_label.grid(row=0, column=4, sticky="w")
            self.split_entry.grid(row=0, column=5, sticky="ew")

    def _browse_csv(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if path:
            self.csv_path.set(path)

    def _browse_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def _read_common_inputs(self) -> tuple[int | None, float, float, str, dict[str, str], str]:
        # Sample size
        sample_size = None
        ss = self.sample_size.get().strip()
        if ss:
            try:
                sample_size = int(ss)
            except Exception:
                raise ValueError("Sample Size must be an integer")

        # Ratios
        try:
            train_ratio = float(self.train_ratio.get())
            test_ratio = float(self.test_ratio.get())
        except Exception:
            raise ValueError("Ratios must be numbers")
        if train_ratio < 0 or test_ratio < 0 or train_ratio + test_ratio > 1.0:
            raise ValueError("Ratios must be >=0 and train+test <= 1.0")

        out_dir = self.output_dir.get().strip() or "."

        mapping: dict[str, str] = {}
        for row in self.mapping_rows:
            key = row["placeholder"].get().strip()
            val = row["column"].get().strip()
            if key:
                mapping[key] = val
        if not mapping:
            raise ValueError("Please add at least one column mapping row")

        template = self.prompt_text.get("1.0", "end").strip()
        if not template:
            raise ValueError("Prompt template cannot be empty")

        return sample_size, train_ratio, test_ratio, out_dir, mapping, template

    def _load_df(self, sample_size: int | None) -> pd.DataFrame:
        if self.dataset_type.get() == "CSV":
            path = self.csv_path.get().strip()
            if not path:
                raise ValueError("Please provide a CSV path")
            return load_dataframe_from_csv(path, sample_size)
        else:
            name = self.hf_name.get().strip()
            split = self.hf_split.get().strip() or "train"
            if not name:
                raise ValueError("Please provide a HuggingFace dataset name")
            return load_dataframe_from_huggingface(name, split, sample_size)

    def _preview(self) -> None:
        try:
            sample_size, _, _, _, mapping, template = self._read_common_inputs()
            df = self._load_df(sample_size=min(3, sample_size or 3))
            records = dataframe_to_text_rows(df, mapping, template)
            preview_text = "\n\n".join(r.get("text", "") for r in records[:3])
            messagebox.showinfo("Preview", preview_text if preview_text else "No preview available")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _run(self) -> None:
        try:
            sample_size, train_ratio, test_ratio, out_dir, mapping, template = self._read_common_inputs()
            df = self._load_df(sample_size)
            records = dataframe_to_text_rows(df, mapping, template)
            train, test, valid = split_records(records, train_ratio, test_ratio)
            save_jsonl(train, f"{out_dir}/train.jsonl")
            save_jsonl(test, f"{out_dir}/test.jsonl")
            save_jsonl(valid, f"{out_dir}/valid.jsonl")
            messagebox.showinfo("Done", f"Saved train/test/valid to {out_dir}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # Mapping management
    def _add_mapping_row(self, placeholder: str | None = None, column: str | None = None) -> None:
        row_dict: dict = {}
        row_index = len(self.mapping_rows) + 1
        lbl = ttk.Label(self.map_rows_container, text=str(row_index))
        var_placeholder = tk.StringVar(value=placeholder or "")
        var_column = tk.StringVar(value=column or "")
        ent_placeholder = ttk.Entry(self.map_rows_container, textvariable=var_placeholder, width=24)
        ent_column = ttk.Entry(self.map_rows_container, textvariable=var_column)
        btn_delete = ttk.Button(self.map_rows_container, text="Delete")

        row_dict["index_label"] = lbl
        row_dict["placeholder"] = var_placeholder
        row_dict["column"] = var_column
        row_dict["entry_placeholder"] = ent_placeholder
        row_dict["entry_column"] = ent_column
        row_dict["btn_delete"] = btn_delete

        self.mapping_rows.append(row_dict)

        # Place widgets inside rows container starting at row 0
        display_row = len(self.mapping_rows) - 1  # 0-based inside container
        lbl.grid(row=display_row, column=0, padx=2, pady=2, sticky="w")
        ent_placeholder.grid(row=display_row, column=1, padx=2, pady=2, sticky="ew")
        ent_column.grid(row=display_row, column=2, padx=2, pady=2, sticky="ew")
        btn_delete.grid(row=display_row, column=3, padx=2, pady=2, sticky="w")

        def do_delete() -> None:
            self._remove_mapping_row(row_dict)
        btn_delete.configure(command=do_delete)

    def _remove_mapping_row(self, row_dict: dict) -> None:
        # Remove widgets from grid
        for key in ["index_label", "entry_placeholder", "entry_column", "btn_delete"]:
            widget = row_dict.get(key)
            if widget is not None:
                widget.grid_forget()
                widget.destroy()
        # Remove from list
        try:
            self.mapping_rows.remove(row_dict)
        except ValueError:
            pass
        # Re-pack remaining rows with correct numbering
        self._reflow_mapping_rows()

    def _reflow_mapping_rows(self) -> None:
        # Reflow inside the rows container starting at row 0
        next_row = 0
        for idx, row in enumerate(self.mapping_rows, start=1):
            row["index_label"].configure(text=str(idx))
            row["index_label"].grid(row=next_row, column=0, padx=2, pady=2, sticky="w")
            row["entry_placeholder"].grid(row=next_row, column=1, padx=2, pady=2, sticky="ew")
            row["entry_column"].grid(row=next_row, column=2, padx=2, pady=2, sticky="ew")
            row["btn_delete"].grid(row=next_row, column=3, padx=2, pady=2, sticky="w")
            next_row += 1

    def _clear_all_mappings(self) -> None:
        while self.mapping_rows:
            self._remove_mapping_row(self.mapping_rows[0])


def main() -> None:
    root = tk.Tk()
    DatasetUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()