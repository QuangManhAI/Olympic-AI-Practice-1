# Bộ đề code tay — Computer Vision

Mục tiêu: tự gõ lại được toàn bộ pipeline phân loại ảnh **không cần LLM, không copy**, và khi gặp lỗi thì biết sửa ở đâu.

**Cách dùng**

- Mỗi phần tạo 1 notebook riêng (`01_data.ipynb`, `02_augment.ipynb`, ...). Tự gõ, không mở lời giải.
- Mỗi bài có mục **Kiểm tra**: chạy các `assert` / con số đó, đúng thì bài đạt.
- Mục **Gợi ý cú pháp** (bấm để mở) chỉ ghi tên hàm/API, không phải lời giải. Chỉ mở khi bí quá 5 phút.
- Đánh dấu `[x]` vào bài đã làm được **2 lần liên tiếp không nhìn gợi ý**.

**Dữ liệu luyện tập**

- Dùng data **ảnh vệ tinh bão** ở `đề code tay/data/data-2/dataset/`:
  - `train/images/` (8385 ảnh `.jpg`, ~960×960 RGB) + `train/annotations.csv`.
  - `public_test/images/` (1200 ảnh) và `private_test/images/` (2398 ảnh), không có nhãn.
- `annotations.csv` có 3 cột: `file_name, is_negative, category_id`.
  - `is_negative = TRUE` → ảnh **không có bão**, `category_id` để trống (NaN). 2548 ảnh.
  - `is_negative = FALSE` → ảnh có bão, `category_id` ∈ **{1, 2, 3, 4}** = cấp cường độ (1 yếu nhất → 4 mạnh nhất).
  - Tên file `YYYYMMDDHHMM_<mã bão>_<hash>.jpg`, vd. `201508010000_1513_8f8e20b8.jpg` = bão số 13 năm 2015, lúc 00:00 ngày 01/08/2015. Mỗi cơn bão có nhiều ảnh cách nhau 3 giờ (110 cơn bão trong train), và cùng một thời điểm thường có 1 ảnh dương + 1 ảnh âm.
  - Train là bão năm 2015–2019; test chủ yếu là bão 2019–2020 → test là **bão mới, chưa thấy khi train**.
- Bài toán quy về **phân loại 5 lớp**: `label = 0` nếu không có bão, `label = category_id` (1–4) nếu có bão.
- File nộp (giả định cùng định dạng `annotations.csv`): `file_name,is_negative,category_id`; dòng âm thì `is_negative=TRUE` và `category_id` để trống. Nếu đề/baseline BTC ghi khác thì theo BTC.
- Khi chỉ cần test shape/code nhanh: `torchvision.datasets.FakeData(size=200, image_size=(3, 224, 224), num_classes=5, transform=...)`.

---

## Phần 0 — Khởi động (5 phút)

- [X] **0.1 Seed và device.** Viết hàm `seed_everything(seed)` đặt seed cho `random`, `numpy`, `torch` (CPU + CUDA). Tạo biến `device` tự chọn `cuda` nếu có, không thì `cpu`.
  - **Kiểm tra:** gọi `seed_everything(42)` hai lần, mỗi lần in `torch.rand(3)` → hai kết quả giống hệt nhau.

<details><summary>Gợi ý cú pháp</summary>

`random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `torch.cuda.is_available()`

</details>

---

## Phần 1 — Dataset và chia tập

- [X] **1.1 Đọc nhãn.** Đọc `train/annotations.csv` bằng pandas. In số ảnh, số ảnh âm/dương, số ảnh mỗi `category_id`.

  - Để ý `is_negative` được đọc thành `bool`, còn `category_id` thành `float64` vì có NaN.
  - Tạo `CLASSES = ['no_storm', 'cat1', 'cat2', 'cat3', 'cat4']`, rồi `idx2class` (`int → tên`) và `class2idx` (`tên → int`) theo **chỉ số PyTorch 0–4** — chỉ số là **số nguyên** `0, 1, 2, 3, 4`, đúng loại giá trị mà `CrossEntropyLoss` nhận làm nhãn, không phải chuỗi `'0'`.
  - Thêm cột `label = category_id.fillna(0).astype(int)` (âm → 0, bão giữ nguyên 1–4). Khi ghi file nộp thì đổi ngược: `label 0` → `is_negative=True`, `category_id` trống; `label k > 0` → `is_negative=False`, `category_id = k`.
  - Thêm cột `storm_id` = phần thứ 2 của tên file (`'1513'`) và `year` = 2 số đầu của `storm_id`.
  - **Kiểm tra:** `len(class2idx) == 5`; `idx2class[class2idx['cat3']] == 'cat3'`; `class2idx['cat3'] == 3`; `idx2class[0] == 'no_storm'`; `df.label.min() == 0 and df.label.max() == 4`; `(df.label == 0).sum() == df.is_negative.sum() == 2548`; số ảnh = 8385, lớp `cat3` ít nhất (770); `df.storm_id.nunique() == 110`.

<details><summary>Gợi ý cú pháp</summary>

`df.category_id.isna()`, `fillna(0).astype(int)`, `df.file_name.str.split('_').str[1]`, `pd.crosstab(df.is_negative, df.label)`

</details>

- [X] **1.2 Chia train/val theo cơn bão (group) + phân tầng.** Chia ~80/20, giữ tỉ lệ lớp, **mọi ảnh của cùng một cơn bão phải nằm cùng một tập**.

  - Vì sao: ảnh cùng cơn bão cách nhau 3 giờ gần như giống hệt nhau. Chia ngẫu nhiên theo ảnh → val bị "lộ" (leak) → F1 val rất cao nhưng test (bão mới) thấp.
  - Thử so sánh: chia ngẫu nhiên `train_test_split(..., stratify=...)` vs chia theo group, train cùng model → F1 val của cách ngẫu nhiên cao hơn hẳn, nhưng nó đang nói dối.
  - **Kiểm tra:** `len(set(tr.storm_id) & set(va.storm_id)) == 0`; `len(set(tr.file_name) & set(va.file_name)) == 0`; tỉ lệ từng lớp ở train và val chênh nhau < 6% (chia theo group không cân đều được như chia theo ảnh).

<details><summary>Gợi ý cú pháp</summary>

`from sklearn.model_selection import StratifiedGroupKFold`; `sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`; `tr_idx, va_idx = next(sgkf.split(df, df.label, groups=df.storm_id))`; `value_counts(normalize=True)`. Cách khác giống test hơn: lấy các cơn bão mới nhất (vd. năm 2019 và cuối 2018) làm val.

</details>

- [X] **1.3 Viết class `ImageDataset(Dataset)`.** Nhận `df`, `img_dir`, `transform`, `has_label=True`. `__getitem__` trả `(tensor, label)` khi có nhãn, `(tensor, file_name)` khi là tập test.
  - Ảnh vẫn nên `.convert('RGB')` (data này toàn JPG RGB, nhưng ngày thi gặp PNG có kênh alpha hoặc ảnh xám sẽ lỗi).
  - Ảnh gốc ~960×960 → đọc chậm. Nếu train nhiều epoch, viết thêm 1 script resize trước toàn bộ ảnh về 256×256, lưu sang thư mục khác.
  - **Kiểm tra:** `x, y = ds[0]` → `x.shape == (3, 224, 224)`, `x.dtype == torch.float32`, `type(y) == int`, `0 <= y <= 4`.

<details><summary>Gợi ý cú pháp</summary>

`from torch.utils.data import Dataset, DataLoader`, `PIL.Image.open(path).convert('RGB')`, `os.path.join`, `df.iloc[i]`

</details>

- [X] **1.4 Dataset cho tập test không có csv.** Viết hàm `make_test_df(img_dir)` liệt kê ảnh bằng `sorted(os.listdir(...))`, chỉ giữ đuôi `.jpg/.jpeg/.png` (cả chữ hoa). Dùng cho cả `public_test` và `private_test`.

  - **Kiểm tra:** `len(make_test_df(public_dir)) == 1200`, `len(make_test_df(private_dir)) == 2398`; thứ tự giữ nguyên giữa 2 lần chạy.
- [ ] **1.5 DataLoader.** Tạo `dl_tr` (shuffle=True, drop_last=True), `dl_va` và `dl_te` (shuffle=False). `num_workers=2`, `pin_memory=True`.

  - **Kiểm tra:** `xb, yb = next(iter(dl_tr))` → `xb.shape == (BS, 3, 224, 224)`, `yb.shape == (BS,)`.
- [ ] **1.6 Xử lý mất cân bằng lớp — 2 cách.**

  - Phân bố 5 lớp: `no_storm` 2548, `cat1` 1931, `cat2` 1583, `cat3` 770, `cat4` 1553.
  - (a) Tính `class_weights` = `N / (num_classes * count_c)` để đưa vào loss. **Kiểm tra (a):** trọng số lớn nhất là của `cat3` (≈ 2.18 trên toàn bộ train).
  - (b) Dùng `WeightedRandomSampler` (mỗi ảnh có trọng số `1 / count` của lớp nó).
  - **Kiểm tra (b):** lấy 20 batch, đếm nhãn → các lớp xấp xỉ đều nhau.

<details><summary>Gợi ý cú pháp</summary>

`torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)`; khi dùng `sampler=` thì **không** truyền `shuffle=True`. `nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float).to(device))`

</details>

- [ ] **1.7 K-Fold theo cơn bão.** Dùng `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` với `groups=df.storm_id` gán cột `fold` vào df.
  - **Kiểm tra:** mỗi ảnh có đúng 1 fold; mỗi `storm_id` chỉ thuộc 1 fold (`df.groupby('storm_id').fold.nunique().max() == 1`); mỗi fold có khoảng 18–25% dữ liệu (không đều tuyệt đối vì các cơn bão dài ngắn khác nhau).

---

## Phần 2 — Augmentation

- [ ] **2.1 Transform bằng torchvision.**
  - `train_tf`: Resize(256) → RandomResizedCrop(224, scale=(0.8, 1)), Rotation(180), ColorJitter nhẹ, ToTensor, Normalize ImageNet.
  - Suy nghĩ trước khi chọn augment cho ảnh vệ tinh:
    - Xoay góc bất kỳ: **hợp lý** (ảnh nhìn từ trên xuống, không có "phía trên").
    - Lật ngang/dọc: bão Bắc bán cầu luôn xoáy **ngược chiều kim đồng hồ** — lật ảnh làm xoáy đổi chiều, ra ảnh không có thật. Thử train có và không có flip rồi so F1 val.
    - Crop quá nhỏ (scale 0.3) có thể cắt mất mắt bão → nhãn sai.
  - `val_tf`: Resize(256) → CenterCrop(224) → ToTensor → Normalize.
  - **Kiểm tra:** cùng 1 ảnh qua `train_tf` 2 lần ra 2 tensor khác nhau; qua `val_tf` 2 lần ra giống hệt.

<details><summary>Gợi ý cú pháp</summary>

`torchvision.transforms as T`, `T.Compose`, `T.Resize`, `T.RandomResizedCrop`, `T.RandomRotation(180)`, `T.RandomHorizontalFlip`, `T.ColorJitter(0.2, 0.2, 0.2)`, `T.ToTensor()`, `T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`

</details>

- [ ] **2.2 Cùng transform đó viết bằng albumentations.** Chú ý albumentations nhận **numpy array**, gọi theo kiểu `tf(image=img)['image']`.
  - **Kiểm tra:** output là tensor `(3, 224, 224)`.

<details><summary>Gợi ý cú pháp</summary>

`import albumentations as A`, `from albumentations.pytorch import ToTensorV2`, `np.array(PIL_img)`, `A.Resize`, `A.RandomResizedCrop`, `A.Rotate(limit=180)`, `A.RandomRotate90`, `A.ColorJitter`, `A.Normalize()`, `ToTensorV2()`

</details>

- [ ] **2.3 Visualize augmentation.** Lấy 1 ảnh, áp `train_tf` 8 lần, vẽ lưới 2×4. Phải **denormalize** trước khi vẽ (nhân std, cộng mean, clip về [0, 1], đổi `(C,H,W)` → `(H,W,C)`).
  - **Kiểm tra:** ảnh hiện màu tự nhiên, không bị tối/nhiễu màu.

<details><summary>Gợi ý cú pháp</summary>

`plt.subplots(2, 4, figsize=(12, 6))`, `x.permute(1, 2, 0)`, `.clamp(0, 1)`, `ax.imshow`, `ax.axis('off')`

</details>

- [ ] **2.4 Tự code augment trên tensor (không dùng thư viện).**

  - `random_rot90(x)`: xoay ngẫu nhiên 0/90/180/270 độ.
  - `random_hflip(x, p)`: lật ngang với xác suất p (để so sánh, xem 2.1).
  - `random_crop(x, size)`: cắt ngẫu nhiên vùng `size×size`.
  - `cutout(x, size)`: tô 0 một ô vuông ngẫu nhiên.
  - **Kiểm tra:** xoay 90° bốn lần ra lại `x` (`torch.equal`); `random_hflip(x, p=1)` lật 2 lần ra lại `x`; `random_crop` ra đúng shape.

<details><summary>Gợi ý cú pháp</summary>

`torch.rot90(x, k, dims=(1, 2))` (với `x` shape `(C,H,W)`), `torch.flip(x, dims=[2])`, `torch.randint(0, 4, (1,)).item()`

</details>
- [ ] **2.5 Mixup.** Viết hàm nhận batch `(x, y)`, trộn với batch đã hoán vị theo `lam ~ Beta(0.4, 0.4)`. Loss = `lam * CE(out, y) + (1 - lam) * CE(out, y_perm)`.

- **Kiểm tra:** với `lam = 1` kết quả bằng loss bình thường.

- [ ] **2.6 (Nâng cao) CutMix.** Dán một vùng chữ nhật từ ảnh khác; `lam` = 1 − diện tích vùng dán / diện tích ảnh.

---

## Phần 3 — Model pretrained có sẵn (quan trọng nhất ngày thi)

- [ ] **3.1 Nạp weights không cần mạng.** Lưu trước weights: `torch.save(torchvision.models.resnet18(weights='DEFAULT').state_dict(), 'r18.pth')`. Sau đó **giả lập ngày thi**: tạo model `weights=None`, nạp từ file.
  - Viết code tự nhận biết file là `state_dict` (dict) hay cả model.
  - **Kiểm tra:** `load_state_dict` báo `<All keys matched successfully>`.

<details><summary>Gợi ý cú pháp</summary>

`torch.load(path, map_location='cpu')`, `isinstance(obj, dict)`, `model.load_state_dict(sd, strict=True)`. Nếu key có tiền tố `module.` → `{k.replace('module.', ''): v for k, v in sd.items()}`

</details>

- [ ] **3.2 Thay head cho từng họ model.** Viết `build_model(name, num_classes, weight_path=None)` hỗ trợ: `resnet18/34`, `densenet121`, `efficientnet_b0/b1/b2`, `mobilenet_v3_large`, `regnet_y_800mf`, `vit_b_16`.
  - **Kiểm tra:** với mọi tên, `build_model(name, 5)(torch.randn(2, 3, 224, 224)).shape == (2, 5)`.

<details><summary>Gợi ý cú pháp</summary>

In `print(model)` rồi nhìn lớp cuối:

- resnet, regnet: `m.fc`
- densenet: `m.classifier` (là `nn.Linear`)
- efficientnet, mobilenet: `m.classifier[-1]`
- vit_b_16 (torchvision): `m.heads.head`

Dùng `.in_features` để lấy số chiều vào.

</details>

- [ ] **3.3 Freeze → unfreeze.** Giai đoạn 1: đóng băng backbone, chỉ train head (2 epoch). Giai đoạn 2: mở hết, LR nhỏ hơn.
  - **Kiểm tra:** in số tham số có `requires_grad=True` ở 2 giai đoạn (giai đoạn 1 chỉ vài nghìn).

<details><summary>Gợi ý cú pháp</summary>

`for p in m.parameters(): p.requires_grad = False`, `sum(p.numel() for p in m.parameters() if p.requires_grad)`

</details>

- [ ] **3.4 LR khác nhau cho backbone và head.** Optimizer với 2 param group: backbone `lr=1e-4`, head `lr=1e-3`.

  - **Kiểm tra:** `[g['lr'] for g in opt.param_groups] == [1e-4, 1e-3]`.
- [ ] **3.5 (Tuỳ chọn) timm nạp weights offline.**

<details><summary>Gợi ý cú pháp</summary>

`timm.create_model('resnet18', pretrained=True, num_classes=6, pretrained_cfg_overlay=dict(file='/path/weights.pth'))`

</details>

---

## Phần 4 — Code model bằng tay

Với mỗi model: (1) đầu vào `(2, 3, 224, 224)` ra `(2, num_classes)`; (2) đếm tham số và so với số chuẩn.

Hàm đếm: `sum(p.numel() for p in m.parameters())`.

- [ ] **4.1 CNN đơn giản.** Khối `ConvBNReLU(in, out)` = Conv3×3 (bias=False) → BatchNorm → ReLU. Xếp 4 tầng (32-64-128-256), mỗi tầng có MaxPool, cuối là AdaptiveAvgPool → Flatten → Linear.

  - **Kiểm tra:** shape đúng; train 3 epoch trên data bão loss giảm.
- [ ] **4.2 ResNet-18.**

  - `BasicBlock`: conv3×3 → BN → ReLU → conv3×3 → BN, cộng shortcut, ReLU. Khi `stride != 1` hoặc đổi số kênh thì shortcut là `downsample = Sequential(conv1×1, BN)`.
  - Thân: conv7×7 stride 2 (64) → BN → ReLU → MaxPool 3×3 stride 2 → 4 stage `[2, 2, 2, 2]` với kênh `64, 128, 256, 512` → AvgPool → fc.
  - **Kiểm tra 1:** với 1000 lớp, số tham số **= 11,689,512**.
  - **Kiểm tra 2 (khó, rất đáng làm):** đặt tên lớp giống torchvision (`conv1, bn1, layer1..layer4, fc`, trong block là `conv1, bn1, conv2, bn2, downsample`) để `my_model.load_state_dict(torchvision.models.resnet18(weights='DEFAULT').state_dict())` khớp hết key, và cho cùng output với model torchvision (`torch.allclose(..., atol=1e-5)` ở chế độ `eval()`).
  - **Mở rộng:** đổi thành ResNet-34 chỉ bằng cách đổi `[3, 4, 6, 3]` → **21,797,672** tham số.
- [ ] **4.3 DenseNet-121.**

  - `DenseLayer`: BN → ReLU → Conv1×1 (4·k kênh) → BN → ReLU → Conv3×3 (k kênh); output **nối (concat)** với input theo chiều kênh. `k = 32` (growth rate).
  - `DenseBlock`: n layer, layer thứ i nhận `in + i·k` kênh.
  - `Transition`: BN → ReLU → Conv1×1 (giảm kênh còn một nửa) → AvgPool 2×2.
  - Thân: Conv7×7 (64) → BN → ReLU → MaxPool → block `[6, 12, 24, 16]` xen 3 Transition → BN cuối → ReLU → AvgPool → Linear. Mọi conv `bias=False`.
  - **Kiểm tra:** 1000 lớp → **7,978,856** tham số; số kênh trước classifier là 1024.
- [ ] **4.4 EfficientNet (khối MBConv).**

  - `SqueezeExcite(ch, se_ch)`: AvgPool → Conv1×1 (ch→se_ch) → SiLU → Conv1×1 (se_ch→ch) → Sigmoid → nhân lại vào input.
  - `MBConv(in, out, expand, kernel, stride)`: Conv1×1 mở rộng (bỏ qua khi expand=1) → BN → SiLU → **Depthwise** Conv k×k (`groups=hidden`) → BN → SiLU → SE (`se_ch = max(1, in // 4)`) → Conv1×1 thu hẹp → BN (không activation). Có residual khi `stride == 1 and in == out`.
  - Cấu hình B0 (expand, kernel, stride, out, số lặp): `(1,3,1,16,1) (6,3,2,24,2) (6,5,2,40,2) (6,3,2,80,3) (6,5,1,112,3) (6,5,2,192,4) (6,3,1,320,1)`. Stem: Conv3×3 stride 2 (32). Head: Conv1×1 → 1280 → AvgPool → Dropout(0.2) → Linear.
  - **Kiểm tra:** shape đúng; số tham số với 1000 lớp ≈ **5.29M** (torchvision: 5,288,548; lệch < 2% là đạt).
- [ ] **4.5 ViT (tự viết attention).**

  - `PatchEmbed`: Conv2d(3, D, kernel=P, stride=P) → flatten → `(B, N, D)` với `N = (224/P)²`.
  - Thêm `cls_token` (`nn.Parameter` shape `(1, 1, D)`) ở đầu và `pos_embed` (`(1, N+1, D)`).
  - `Attention` **tự viết, không dùng `nn.MultiheadAttention`**: `qkv = Linear(D, 3D)` → tách heads `(B, H, N, d)` → `softmax(Q·Kᵀ / √d) · V` → gộp heads → `Linear(D, D)`.
  - `Block` (pre-norm): `x = x + Attn(LN(x))`; `x = x + MLP(LN(x))`, MLP = Linear(D, 4D) → GELU → Linear(4D, D).
  - Cuối: LN → lấy token `cls` → Linear.
  - **Kiểm tra 1:** cấu hình ViT-Tiny (P=16, D=192, depth=12, heads=3), 1000 lớp → ≈ **5.72M** tham số.
  - **Kiểm tra 2:** attention của bạn cho cùng kết quả với `F.scaled_dot_product_attention(q, k, v)` trên cùng q, k, v.

<details><summary>Gợi ý cú pháp</summary>

`x.flatten(2).transpose(1, 2)`, `self.cls.expand(B, -1, -1)`, `torch.cat([cls, x], dim=1)`, `qkv.reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)`, `q @ k.transpose(-2, -1)`, `attn.softmax(dim=-1)`, `nn.init.trunc_normal_(p, std=0.02)`

</details>

---

## Phần 5 — Optimizer và LR scheduler

- [ ] **5.1 Ba optimizer.** `SGD(momentum=0.9, nesterov=True)`, `Adam`, `AdamW(weight_decay=0.05)`. Train CNN 4.1 mỗi loại 3 epoch, ghi lại loss.
- [ ] **5.2 Không weight decay cho bias và BN/LN.** Tách tham số: `p.ndim <= 1` → nhóm `weight_decay=0`.

  - **Kiểm tra:** tổng số tham số 2 nhóm = tổng tham số model.
- [ ] **5.3 Scheduler có sẵn.** Vẽ đường LR theo step (không cần train, chỉ gọi `opt.step(); sched.step()` trong vòng lặp) cho:

  - `StepLR(step_size=10, gamma=0.1)`
  - `CosineAnnealingLR(T_max=total_steps)`
  - `OneCycleLR(max_lr=1e-3, total_steps=total_steps)`
  - `ReduceLROnPlateau(mode='max', patience=2)` — scheduler này gọi `sched.step(val_f1)`.
  - **Kiểm tra:** hình vẽ đúng dạng (bậc thang, nửa cos, lên rồi xuống).
- [ ] **5.4 Tự code warmup + cosine.** Viết `lr_lambda(step)`: tăng tuyến tính từ 0 đến 1 trong `warmup` step, sau đó giảm theo cosine về `min_ratio`. Dùng với `LambdaLR`.

  - **Kiểm tra:** `lr_lambda(0) == 0`, `lr_lambda(warmup) == 1`, `lr_lambda(total) ≈ min_ratio`; vẽ ra đúng hình.

<details><summary>Gợi ý cú pháp</summary>

`torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)`, `math.cos(math.pi * progress)`, `opt.param_groups[0]['lr']`. Scheduler theo step thì gọi `sched.step()` **sau mỗi batch**; theo epoch thì sau mỗi epoch — đừng nhầm.

</details>

- [ ] **5.5 LR finder đơn giản.** Tăng LR theo cấp số nhân từ 1e-7 đến 1 trong 100 batch, ghi loss, vẽ log-scale. Chọn LR ≈ 1/10 điểm loss thấp nhất.

---

## Phần 6 — Training loop

- [ ] **6.1 Hai hàm `train_one_epoch(model, dl, opt, crit, sched=None)` và `evaluate(model, dl)`.**

  - Train: `model.train()`, đưa lên device, `zero_grad` → forward → loss → backward → step. Trả loss trung bình.
  - Evaluate: `model.eval()` + `torch.no_grad()`, trả `loss, acc, macro_f1`, cùng mảng xác suất.
  - **Kiểm tra:** model chưa train có macro F1 thấp (cỡ ≤ 0.2, luôn đoán 1–2 lớp); sau 3 epoch F1 tăng rõ. Mốc tham chiếu: đoán toàn `no_storm` được acc ≈ 0.30 — model phải vượt xa con số này.
- [ ] **6.2 Mixed precision + gradient clipping.** Thêm `torch.autocast('cuda', dtype=torch.float16)` và `GradScaler`; clip norm = 1.0.

  - **Kiểm tra:** thời gian 1 epoch giảm so với 6.1; loss không ra `nan`.

<details><summary>Gợi ý cú pháp</summary>

`scaler = torch.amp.GradScaler()`, `scaler.scale(loss).backward()`, `scaler.unscale_(opt)`, `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`, `scaler.step(opt)`, `scaler.update()`. Với bfloat16 thì không cần scaler.

</details>

- [ ] **6.3 Vòng `fit` hoàn chỉnh.** Lưu `best.pt` theo **macro F1 trên val**, early stopping (patience=5), lưu lịch sử `history = {'tr_loss': [], 'va_loss': [], 'va_f1': []}`, in thời gian mỗi epoch.
- [ ] **6.4 Lưu và resume checkpoint.** Lưu `{'model', 'opt', 'sched', 'epoch', 'best_f1'}`. Tắt kernel giữa chừng → chạy lại tiếp từ epoch đó.

  - **Kiểm tra:** sau resume, epoch tiếp theo in ra đúng số thứ tự, LR tiếp tục đúng chỗ.
- [ ] **6.5 Predict + TTA + file nộp.**

  - `predict(model, dl_te, tta=True)`: softmax của ảnh gốc + 3 bản xoay 90/180/270°, lấy trung bình (không dùng lật, xem 2.1).
  - Viết `to_submission(names, preds)`: `pred 0` → `is_negative=True`, `category_id` trống; `pred k > 0` → `is_negative=False`, `category_id=k`. Ghi `submission.csv` (`file_name,is_negative,category_id`) cho **cả public_test và private_test** (gộp hoặc tách file theo yêu cầu BTC), kiểm tra bằng `check_submission`, rồi zip kèm notebook.
  - `check_submission`: đúng tên cột, đủ số dòng, không trùng, tập tên file khớp thư mục test, `category_id` ∈ {1,2,3,4}, `category_id` trống **khi và chỉ khi** `is_negative` là True, in phân bố nhãn.
  - **Kiểm tra:** cố tình xoá 1 dòng / sửa 1 `category_id` thành 9 / gán `category_id=2` cho 1 dòng âm → hàm đều báo lỗi.

<details><summary>Gợi ý cú pháp</summary>

`torch.rot90(x, k, dims=(2, 3))` (batch `(B,C,H,W)`), `np.where(preds == 0, np.nan, preds)`, `df['category_id'] = df['category_id'].astype('Int64')` để ghi số nguyên mà vẫn để trống được (không ra `2.0`), `df.to_csv(..., index=False)`

</details>
- [ ] **6.6 Ensemble.** Train 2 model khác nhau (vd. efficientnet_b0 + resnet34), lấy trung bình xác suất.

- **Kiểm tra:** F1 ensemble trên val ≥ model tốt hơn trong 2 model (thường là vậy).

- [ ] **6.7 (Nâng cao) EMA của trọng số.** Sau mỗi step: `ema_p = 0.999·ema_p + 0.001·p`. Đánh giá bằng model EMA.

---

## Phần 7 — Visualize

- [ ] **7.1 Đường cong train.** Vẽ `tr_loss`/`va_loss` và `va_f1` theo epoch trên 2 subplot. Tự nhận biết overfit (val loss tăng khi train loss giảm).
- [ ] **7.2 Phân bố lớp.** Bar chart số ảnh mỗi lớp ở train; và phân bố nhãn dự đoán trên test.
- [ ] **7.3 Confusion matrix** trên val, có tên lớp, dạng heatmap. In `classification_report`.
  - Cấp bão là nhãn **có thứ tự** (1 < 2 < 3 < 4): xem lỗi có dồn vào các ô sát đường chéo không (nhầm cat2↔cat3 dễ hiểu hơn nhầm cat1↔cat4). Tách riêng: acc nhị phân "có bão / không bão" và macro F1 trên 4 cấp bão.

<details><summary>Gợi ý cú pháp</summary>

`sklearn.metrics.confusion_matrix`, `classification_report(y, p, target_names=...)`, `ConfusionMatrixDisplay(cm, display_labels=...).plot()` hoặc `seaborn.heatmap(cm, annot=True, fmt='d')`

</details>

- [ ] **7.4 Ảnh dự đoán sai.** Vẽ 16 ảnh val bị sai nhiều nhất (xác suất lớp đúng thấp nhất), tiêu đề `"thật: X / đoán: Y (0.87)"`.
- [ ] **7.5 (Nâng cao) Grad-CAM** cho ResNet: dùng `register_forward_hook` / `register_full_backward_hook` lên `layer4`, lấy trung bình gradient làm trọng số cho feature map, ReLU, resize lên 224, chồng lên ảnh.

---

## Phần 8 — Tìm và sửa lỗi (luyện "vào thi biết sửa")

Mỗi đoạn dưới đây có **ít nhất 1 lỗi**. Đọc, đoán lỗi, rồi chạy để xác nhận.

**8.1**

```python
for x, y in dl_tr:
    out = model(x.to(device))
    loss = crit(out, y.to(device))
    loss.backward()
    opt.step()
```

**8.2**

```python
model.train()
with torch.no_grad():
    for x, y in dl_va:
        preds += model(x.to(device)).argmax(1).tolist()
```

**8.3**

```python
class Net(nn.Module):
    def forward(self, x):
        ...
        return torch.softmax(self.fc(x), dim=1)
crit = nn.CrossEntropyLoss()
```

**8.4**

```python
dl_te = DataLoader(test_ds, batch_size=64, shuffle=True)
names = sorted(os.listdir(test_dir))
preds = predict(model, dl_te)
pd.DataFrame({'file_name': names, 'is_negative': preds == 0,
              'category_id': preds}).to_csv('submission.csv')
```

**8.5**

```python
val_tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])          # train_tf có Normalize
img = Image.open(path)                                             # ngày thi có thể gặp PNG RGBA
label = int(row['category_id']) - 1
```

**8.6**

```python
for ep in range(EPOCHS):
    train_one_epoch(model, dl_tr, opt, crit)
    sched.step()          # sched = OneCycleLR(..., total_steps=EPOCHS * len(dl_tr))
```

**8.7**

```python
m = torchvision.models.efficientnet_b0()
m.fc = nn.Linear(1280, 5)
m.load_state_dict(torch.load('effb0.pth'))
```

**8.8** — `RuntimeError: CUDA out of memory` khi batch_size=128, ảnh 384. Liệt kê 4 cách xử lý theo thứ tự nên thử.

<details><summary>Đáp án (chỉ mở sau khi tự làm)</summary>

- **8.1** Thiếu `opt.zero_grad()` → gradient cộng dồn qua các batch.
- **8.2** Phải `model.eval()` (BN/Dropout đang ở chế độ train); `preds` chưa khởi tạo.
- **8.3** `CrossEntropyLoss` đã có softmax bên trong → trả **logits**, không softmax.
- **8.4** Test loader `shuffle=True` → dự đoán lệch thứ tự với `names`; thứ tự trong dataset phải cùng cách `sorted` với `names`; thiếu `index=False`; dòng âm vẫn ghi `category_id = 0` thay vì để trống, và dòng dương ghi ra `2.0` nếu cột bị ép float (dùng `astype('Int64')`).
- **8.5** Val thiếu `Normalize` giống train; thiếu `.convert('RGB')`; `int(NaN)` báo lỗi với ảnh âm (`category_id` trống); `- 1` làm lệch nhãn — ở bộ này `label = category_id.fillna(0)`, lớp 0 là `no_storm`, không trừ 1 (đọc kỹ đề/baseline xem nhãn bắt đầu từ đâu).
- **8.6** OneCycleLR tính theo **step** → phải gọi `sched.step()` sau mỗi batch, gọi theo epoch thì cả quá trình train chỉ đi được vài step đầu của chu kỳ, LR gần như đứng yên ở mức khởi đầu rất nhỏ.
- **8.7** EfficientNet không có `fc` — head là `m.classifier[-1]`; và phải **nạp weights trước, thay head sau** (weights gốc có 1000 lớp).
- **8.8** Giảm batch size → dùng autocast fp16/bf16 → giảm kích thước ảnh → model nhỏ hơn / gradient accumulation. Sau khi lỗi nhớ `del` biến và `torch.cuda.empty_cache()` (hoặc restart kernel).

</details>

---

## Phần 9 — Bài thi thử tổng hợp (bấm giờ 90 phút, không nhìn notebook cũ)

Notebook trắng, dữ liệu ảnh vệ tinh bão (`data-2/dataset`). Phải xong theo đúng thứ tự:

1. (10') Đọc `annotations.csv`, tạo `label` 0–4 và `storm_id`, chia `StratifiedGroupKFold` theo cơn bão (~80/20), Dataset + DataLoader, vẽ 8 ảnh kèm nhãn.
2. (10') `build_model('efficientnet_b0', 5)` nạp weights từ file local.
3. (40') Train AdamW + warmup cosine + AMP, augment xoay (không lật), lưu best theo macro F1, vẽ đường cong.
4. (15') Confusion matrix, predict `public_test` + `private_test` với TTA xoay, ghi `submission.csv` đúng định dạng `file_name,is_negative,category_id`, chạy `check_submission`, zip.
5. (15') Đổi sang `resnet34`, train lại, ensemble 2 model, so F1 val.

**Đạt:** xong cả 5 bước trong 90 phút và macro F1 val vượt `SimpleCNN` (bài 4.1) rõ rệt. Làm lại đến khi đạt 2 lần liên tiếp.

---

## Bảng cú pháp hay quên

| Việc                                     | Cú pháp                                                                                    |
| ----------------------------------------- | -------------------------------------------------------------------------------------------- |
| Đổi `(C,H,W)` → `(H,W,C)` để vẽ | `x.permute(1, 2, 0).numpy()`                                                               |
| Ghép batch hai chiều                    | `torch.cat([a, b], dim=0)`                                                                 |
| Lật ngang tensor batch                   | `torch.flip(x, dims=[3])`                                                                  |
| Xác suất                                | `logits.softmax(dim=1)`                                                                    |
| Đưa về numpy                           | `t.detach().cpu().numpy()`                                                                 |
| Macro F1                                  | `f1_score(y, p, average='macro')`                                                          |
| Đếm tham số                            | `sum(p.numel() for p in m.parameters())`                                                   |
| Lưu / nạp                               | `torch.save(m.state_dict(), p)` / `m.load_state_dict(torch.load(p, map_location='cpu'))` |
| Tắt gradient                             | `with torch.no_grad():` / `p.requires_grad = False`                                      |
| LR hiện tại                             | `opt.param_groups[0]['lr']`                                                                |
| Ghi csv                                   | `df.to_csv('submission.csv', index=False)`                                                 |
| Zip                                       | `zipfile.ZipFile('s.zip', 'w').write('submission.csv')`                                    |
| Giải nén có mật khẩu                 | `zipfile.ZipFile(p).extractall(out, pwd=b'...')` hoặc `!unzip -P pass p -d out`         |
