import argparse
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
import streamlit as st

LABEL_NAME = {0: "clear", 1: "hyphae", 2: "spor"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", type=str, required=True, help="Path to HDF5 with datasets: images, labels")
    p.add_argument("--index_csv", type=str, required=True, help="CSV with column 'idx' listing indices to review")
    p.add_argument("--log_csv", type=str, default=None, help="Where to log edits (CSV). Default: alongside index_csv")
    p.add_argument("--image_key", type=str, default="images")
    p.add_argument("--label_key", type=str, default="labels")
    return p.parse_args()


def ensure_rgb_uint8(img):
    img = np.asarray(img)
    # HWC
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        # common if stored as float [0,1] or [0,255]
        # do a conservative conversion
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
    # ensure 3 channels
    if img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img


def main():
    args = parse_args()
    hdf5_path = Path(args.hdf5)
    index_csv = Path(args.index_csv)
    log_csv = Path(args.log_csv) if args.log_csv else index_csv.with_name(index_csv.stem + "_edits_log.csv")

    st.set_page_config(page_title="HDF5 Label Reviewer", layout="wide")
    st.title("HDF5 Mislabel Reviewer (edits labels in place)")

    # --- Load review indices ---
    df_idx = pd.read_csv(index_csv)
    if "idx" not in df_idx.columns:
        st.error(f"index_csv must contain a column named 'idx'. Got columns: {list(df_idx.columns)}")
        st.stop()

    review_indices = df_idx["idx"].astype(int).tolist()
    if len(review_indices) == 0:
        st.warning("No indices found in index_csv.")
        st.stop()

    # Optional extra columns (if you saved them)
    extra_cols = [c for c in df_idx.columns if c != "idx"]

    # --- Session state ---
    if "pos" not in st.session_state:
        st.session_state.pos = 0
    if "history" not in st.session_state:
        st.session_state.history = []  # stack of (idx, old_label, new_label, timestamp)
    if "filter_mode" not in st.session_state:
        st.session_state.filter_mode = "all"

    # --- Safety note + backup hint ---
    st.info(
        "This tool edits labels *in place* in the HDF5 file. "
        "Consider making a backup copy of the HDF5 before a big review pass."
    )

    # --- Controls ---
    colA, colB, colC, colD = st.columns([1.2, 1.2, 1.2, 2.4])
    with colA:
        jump = st.number_input("Jump to position", min_value=0, max_value=len(review_indices) - 1,
                               value=st.session_state.pos, step=1)
        if st.button("Go"):
            st.session_state.pos = int(jump)
    with colB:
        if st.button("Prev"):
            st.session_state.pos = max(0, st.session_state.pos - 1)
    with colC:
        if st.button("Next"):
            st.session_state.pos = min(len(review_indices) - 1, st.session_state.pos + 1)
    with colD:
        st.write(f"Review item **{st.session_state.pos + 1} / {len(review_indices)}**")

    idx = review_indices[st.session_state.pos]

    # --- Load current image/label from HDF5 ---
    with h5py.File(hdf5_path, "r+") as f:
        if args.image_key not in f or args.label_key not in f:
            st.error(f"HDF5 must contain datasets '{args.image_key}' and '{args.label_key}'. Keys: {list(f.keys())}")
            st.stop()

        images = f[args.image_key]
        labels = f[args.label_key]

        if idx < 0 or idx >= len(labels):
            st.error(f"Index {idx} out of range for labels length {len(labels)}")
            st.stop()

        img = ensure_rgb_uint8(images[idx])
        cur_label = int(labels[idx])

        left, right = st.columns([2.2, 1.4])

        with left:
            st.image(img, caption=f"HDF5 idx={idx}", use_container_width=True)

        with right:
            st.subheader("Current label")
            st.write(f"**{cur_label}** = {LABEL_NAME.get(cur_label, 'unknown')}")

            # show any extra info from CSV if present (pred, probs, etc.)
            if extra_cols:
                row = df_idx.iloc[st.session_state.pos].to_dict()
                st.subheader("Context (from index_csv)")
                for k in extra_cols:
                    st.write(f"- **{k}**: {row.get(k)}")

            st.subheader("Relabel")
            c1, c2, c3 = st.columns(3)
            # Use buttons for speed
            if c1.button("0: clear"):
                new_label = 0
                old_label = int(labels[idx])
                labels[idx] = new_label
                f.flush()  # <-- ADD THIS

                st.session_state.history.append((idx, old_label, new_label, time.time()))
                append_log(log_csv, idx, old_label, new_label)
                st.session_state.pos = min(len(review_indices) - 1, st.session_state.pos + 1)
                st.rerun()

            if c2.button("1: hyphae"):
                new_label = 1
                old_label = int(labels[idx])
                labels[idx] = new_label
                f.flush()  # <-- ADD THIS

                st.session_state.history.append((idx, old_label, new_label, time.time()))
                append_log(log_csv, idx, old_label, new_label)
                st.session_state.pos = min(len(review_indices) - 1, st.session_state.pos + 1)
                st.rerun()

            if c3.button("2: spor"):
                new_label = 2
                old_label = int(labels[idx])
                labels[idx] = new_label
                f.flush()  # <-- ADD THIS
                st.session_state.history.append((idx, old_label, new_label, time.time()))
                append_log(log_csv, idx, old_label, new_label)
                st.session_state.pos = min(len(review_indices) - 1, st.session_state.pos + 1)
                st.rerun()

            st.divider()
            if st.button("Skip (no change)"):
                st.session_state.pos = min(len(review_indices) - 1, st.session_state.pos + 1)
                st.rerun()

            if st.button("Undo last change"):
                if not st.session_state.history:
                    st.warning("Nothing to undo yet.")
                else:
                    last_idx, old_label, new_label, ts = st.session_state.history.pop()
                    labels[last_idx] = old_label
                    append_log(log_csv, last_idx, new_label, old_label, undo=True)
                    st.success(f"Undid: idx={last_idx} {new_label} -> {old_label}")
                    # jump back to that item if it's in your review list
                    if last_idx in review_indices:
                        st.session_state.pos = review_indices.index(last_idx)
                    st.rerun()

    st.caption(f"Edits log: {log_csv}")


def append_log(log_csv: Path, idx: int, old_label: int, new_label: int, undo: bool = False):
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "idx": int(idx),
        "old_label": int(old_label),
        "new_label": int(new_label),
        "undo": bool(undo),
    }
    df = pd.DataFrame([row])
    if log_csv.exists():
        df.to_csv(log_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(log_csv, index=False)


if __name__ == "__main__":
    main()


 # streamlit run ./common/review_hdf5_labels.py --   --hdf5 "C:\Users\Intel User\Desktop\blackbird_scripts\test_1_26_26_3_class_powdery.hdf5"   --index_csv "C:\Users\Intel User\Desktop\blackbird_scripts\results\journal\inference_results\baseline\ResNet\review_indices.csv"
