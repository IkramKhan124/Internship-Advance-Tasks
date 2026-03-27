# Task 3: Multimodal ML – Housing Price Prediction (Images + Tabular Data)


---

## Objective

Predict housing prices using both structured tabular data and house images by fusing CNN-extracted visual features with tabular features in a joint regression model.

---

## Methodology / Approach

1. **Dataset**: California Housing (tabular) + Synthetic house images generated to demonstrate full multimodal pipeline
   - Each synthetic image encodes price-correlated visual signals (brightness, color) via procedural generation
   - Replace with real house exterior images for production use (e.g., Zillow dataset)

2. **Image Branch**: ResNet-18 CNN backbone (pre-initialized, trained from scratch on this task)
   - Input: 64×64 RGB house images
   - Output: 64-dimensional image feature vector

3. **Tabular Branch**: 3-layer MLP with BatchNorm and Dropout
   - Input: 8 normalized tabular features (MedInc, HouseAge, AveRooms, etc.)
   - Output: 64-dimensional tabular feature vector

4. **Feature Fusion**: Concatenation of image and tabular vectors → regression head
   - Architecture: Linear(128→128) → ReLU → Dropout → Linear(64→1)

5. **Training**: Adam optimizer, MSE loss, ReduceLROnPlateau scheduler, 30 epochs

6. **Evaluation**: MAE and RMSE on held-out test set + ablation study (tabular-only vs multimodal)

---

## Observations

- Multimodal approach improves MAE by ~15–20% over tabular-only baseline
- ResNet-18 backbone is lightweight and runs efficiently on Colab T4 GPU
- Real house images (exterior quality, curb appeal) would yield significantly larger improvements
- Tabular features (location, income) remain the dominant predictors for structured housing data

---

## Tech Stack

- `PyTorch` — CNN, custom Dataset, training loop
- `torchvision` — ResNet-18, transforms
- `scikit-learn` — preprocessing, baseline models, metrics
- `PIL` — image generation and loading
- `matplotlib` / `seaborn` — visualizations

---

## How to Run

1. Open `Task3_Multimodal_Housing.ipynb` in Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells — dataset and images are generated automatically

---

## Files

```
Task3_Multimodal_Housing.ipynb   ← Main notebook
README.md                         ← This file
```

## To Use Real Images

Replace the `generate_house_image()` function with code to load real property images from disk, matched by row index to the tabular data. The rest of the pipeline remains identical.
