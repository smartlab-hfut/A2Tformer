# A2Tformer

**A2Tformer** is a Transformer-based deep learning model for non-stationary time series classification.  
It introduces a novel autocorrelation-based attention mechanism (A²T) and a Learnable Wavelet Attention module to enhance robustness.

---

## 🧠 Core Features

- 📌 A²T: attention based on autocorrelation function, replacing positional encoding
- 🔁 Dual-branch design: frequency-aware wavelet attention + time-domain attention
- 🧪 Strong performance on UCR archive with fewer parameters

---

## 📁 Project Structure

```bash
├── main.py              # Main training script
├── model.py             # Transformer model with A²T and PWTM
├── dataprosess.py       # Data loading and normalization
├── test_model.py        # Evaluation and plotting
├── config.json          # Dataset configurations (optional)
├── requirements.txt     # Dependencies
├── train_ucr.sh         # Train all datasets
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/your-name/A2Tformer.git
cd A2Tformer
pip install -r requirements.txt
bash train_ucr.sh
```

---

## 📊 Datasets

The model supports all UCR 2018 datasets. Please place the dataset under:

```plaintext
./data/UCRArchive_2018/<DatasetName>/<DatasetName>_TRAIN.tsv
./data/UCRArchive_2018/<DatasetName>/<DatasetName>_TEST.tsv
```

---

## 📈 Visualization

You may enable feature visualization in `main.py` and generate:

- `viewdata/*.npy`: feature tensors before & after FFN
- `plots/*.png`: training curves
- `reports/*.csv`: classification reports

---

## 📄 License

MIT License.

---

## 📬 Contact

Feel free to reach out for collaborations or questions.

- 📧 luohuan@mail.hfut.edu.cn
- 📧 liqiyue@mail.ustc.edu.cn
