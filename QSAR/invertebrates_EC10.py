import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
from transformers import BertTokenizerFast, BertModel
import os
from sklearn.preprocessing import OneHotEncoder

# ========== 数据读取 ==========
train_data = pd.read_excel("./invertebrates_EC10_train.xlsx")
train_data = train_data.dropna().copy()

test_data = pd.read_excel("./invertebrates_EC10_test.xlsx")
test_data = test_data.dropna().copy()

# ========== 训练集 ==========
train_smiles = train_data['SMILES_Canonical_RDKit'].tolist()
train_labels = train_data['mgperL'].values.astype(np.float32)
train_labels = np.log1p(train_labels)

# ========== 测试集 ==========
test_smiles = test_data['SMILES_Canonical_RDKit'].tolist()
test_labels = test_data['mgperL'].values.astype(np.float32)
test_labels = np.log1p(test_labels)

# ========== One-Hot 编码辅助 ==========
def fit_and_transform_column(train_df, test_df, column_name):
    """
    在训练集上 fit，再对 train/test 一起 transform，保证维度一致
    """
    train_unique = train_df[column_name].dropna().unique()
    test_unique = test_df[column_name].dropna().unique()
    all_unique = np.unique(np.concatenate([train_unique, test_unique]))

    if len(all_unique) > 1:
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        enc.fit(train_df[[column_name]])
        train_encoded = enc.transform(train_df[[column_name]])
        test_encoded = enc.transform(test_df[[column_name]])
        return train_encoded, test_encoded
    else:
        return None, None

# ========== 编码 effect、endpoint、species_group ==========
effect_train, effect_test = fit_and_transform_column(train_data, test_data, 'effect')
endpoint_train, endpoint_test = fit_and_transform_column(train_data, test_data, 'endpoint')
species_train, species_test = fit_and_transform_column(train_data, test_data, 'species_group')

# ========== 拼接额外特征 ==========
train_extra_features = train_data['Duration_Value'].values.reshape(-1, 1).astype(np.float32)
test_extra_features  = test_data['Duration_Value'].values.reshape(-1, 1).astype(np.float32)

for tr_enc, te_enc in [
    (effect_train, effect_test),
    (endpoint_train, endpoint_test),
    (species_train, species_test)
]:
    if tr_enc is not None:
        train_extra_features = np.hstack((train_extra_features, tr_enc.astype(np.float32)))
        test_extra_features  = np.hstack((test_extra_features, te_enc.astype(np.float32)))

extra_dim = train_extra_features.shape[1]

# ========== 设备 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 数据集 ==========
class SMILES_Dataset(Dataset):
    def __init__(self, smiles, reg_labels, extra_features=None, tokenizer=None, max_length=128):
        self.smiles = smiles
        self.reg_labels = reg_labels
        self.extra_features = extra_features
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        s = self.smiles[idx]
        reg_label = self.reg_labels[idx]
        tokens = self.tokenizer(
            s,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        tokens = {key: val.squeeze(0) for key, val in tokens.items()}

        if self.extra_features is not None:
            extra_feat = torch.tensor(self.extra_features[idx], dtype=torch.float32)
            return tokens, torch.tensor(reg_label, dtype=torch.float32), extra_feat
        else:
            return tokens, torch.tensor(reg_label, dtype=torch.float32)

# ========== 加载本地 BERT ==========
checkpoint = '../BERT'
tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
bert_model = BertModel.from_pretrained(checkpoint)

# ========== 模型 ==========
class Bert_Regression(nn.Module):
    def __init__(self, dropout_rate, fc1_size, fc2_size, fc3_size, extra_dim=0):
        super(Bert_Regression, self).__init__()
        self.bert = bert_model
        hidden_size = self.bert.config.hidden_size
        input_dim = hidden_size + extra_dim

        self.regressor = nn.Sequential(
            nn.Linear(input_dim, fc1_size),
            nn.ReLU(),
            nn.BatchNorm1d(fc1_size),
            nn.Dropout(dropout_rate),

            nn.Linear(fc1_size, fc2_size),
            nn.ReLU(),
            nn.BatchNorm1d(fc2_size),
            nn.Dropout(dropout_rate),

            nn.Linear(fc2_size, fc3_size),
            nn.ReLU(),
            nn.BatchNorm1d(fc3_size),
            nn.Dropout(dropout_rate),

            nn.Linear(fc3_size, 1),
            nn.Softplus()
        )

    def forward(self, tokens, extra_features=None):
        outputs = self.bert(**tokens)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        if extra_features is not None:
            x = torch.cat([cls_embedding, extra_features], dim=1)
        else:
            x = cls_embedding
        reg_output = self.regressor(x).squeeze(-1)
        return reg_output

# ========== 超参数 ==========
best_params = {
    'dropout_rate': 0.21357344690077285,
    'fc1_size': 896,
    'fc2_size': 352,
    'fc3_size': 192,
    'learning_rate': 1.885835543930883e-05,
    'weight_decay': 0.040884771694787714
}

best_dropout = best_params['dropout_rate']
best_fc1 = best_params['fc1_size']
best_fc2 = best_params['fc2_size']
best_fc3 = best_params['fc3_size']
best_lr = best_params['learning_rate']
best_wd = best_params['weight_decay']

# ========== DataLoader ==========
train_dataset = SMILES_Dataset(
    train_smiles, train_labels,
    extra_features=train_extra_features,
    tokenizer=tokenizer
)

test_dataset = SMILES_Dataset(
    test_smiles, test_labels,
    extra_features=test_extra_features,
    tokenizer=tokenizer
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# ========== 初始化模型 & 训练 ==========
model = Bert_Regression(best_dropout, best_fc1, best_fc2, best_fc3, extra_dim=extra_dim).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)
criterion = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

for epoch in range(30):
    model.train()
    train_losses = []

    for tokens, labels_tensor, extra_feats in train_loader:
        tokens = {k: v.to(device) for k, v in tokens.items()}
        labels_tensor = labels_tensor.to(device)
        extra_feats = extra_feats.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(tokens, extra_feats)
            loss = criterion(outputs, labels_tensor)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_losses.append(loss.item())

    # ===== 测试集评估 =====
    model.eval()
    preds_all, targets_all = [], []

    with torch.no_grad():
        for tokens, labels_tensor, extra_feats in test_loader:
            tokens = {k: v.to(device) for k, v in tokens.items()}
            labels_tensor = labels_tensor.to(device)
            extra_feats = extra_feats.to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(tokens, extra_feats)

            preds_all.append(outputs.detach().cpu().numpy())
            targets_all.append(labels_tensor.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    rmse = np.sqrt(np.mean((preds_all - targets_all) ** 2))
    mae  = mean_absolute_error(targets_all, preds_all)
    r2   = r2_score(targets_all, preds_all)

    print(f"Epoch {epoch+1:02d} | RMSE {rmse:.4f} | MAE {mae:.4f} | R2 {r2:.4f}")

# ========== 保存训练后的模型 ==========
os.makedirs("./saved_models", exist_ok=True)
torch.save(model.state_dict(), "./saved_models/invertebrates_EC10_train_test_split.pth")
print("✅ 训练完成，模型已保存到 ./saved_models/invertebrates_EC10_train_test_split.pth")

# ========== 提取嵌入并保存 ==========
def get_embeddings_for_loader(model, loader):
    all_embeddings = []
    all_extra_features = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for tokens, labels_tensor, extra_feats in loader:
            tokens = {key: val.to(device) for key, val in tokens.items()}
            extra_feats = extra_feats.to(device)

            outputs = model.bert(**tokens)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, 768]

            all_embeddings.append(cls_embeddings.cpu().numpy())
            all_extra_features.append(extra_feats.cpu().numpy())
            all_labels.append(labels_tensor.numpy())

    return (
        np.vstack(all_embeddings),
        np.vstack(all_extra_features),
        np.concatenate(all_labels)
    )

smiles_emb_train, extra_feat_train, labels_train = get_embeddings_for_loader(model, train_loader)
smiles_emb_test,  extra_feat_test,  labels_test  = get_embeddings_for_loader(model, test_loader)

X_train = np.hstack([smiles_emb_train, extra_feat_train])
X_test  = np.hstack([smiles_emb_test,  extra_feat_test])

os.makedirs('./invertebrates_EC10/embeddings', exist_ok=True)
os.makedirs('./invertebrates_EC10/bert_prediction', exist_ok=True)

np.save('./invertebrates_EC10/embeddings/train_single.npy', X_train)
np.save('./invertebrates_EC10/embeddings/test_single.npy', X_test)
np.save('./invertebrates_EC10/embeddings/train_labels_single.npy', labels_train)
np.save('./invertebrates_EC10/embeddings/test_labels_single.npy', labels_test)

# ========== 测试集预测并保存 ==========
y_preds = []
model.eval()

with torch.no_grad():
    for tokens, labels_tensor, extra_feats in test_loader:
        tokens = {k: v.to(device) for k, v in tokens.items()}
        extra_feats = extra_feats.to(device)
        outputs = model(tokens, extra_feats)
        y_preds.extend(outputs.squeeze().cpu().numpy())

y_preds = np.array(y_preds)
np.save('./invertebrates_EC10/bert_prediction/bert_single_test_predictions.npy', y_preds)

print("✅ 嵌入、标签与测试集预测已保存（train/test split 版本）。")
import os
import numpy as np
from xgboost import XGBRegressor

# =========================================================
# 1. 目录
# =========================================================
data_dir = './invertebrates_EC10/embeddings'
save_dir = './invertebrates_EC10/xgb_predictions'
os.makedirs(save_dir, exist_ok=True)

# =========================================================
# 2. 最优超参数（保持不变）
# =========================================================
best_params = {
    'n_estimators': 500,
    'learning_rate': 0.014509456908913889,
    'max_depth': 5,
    'subsample': 0.9387632330805947,
    'colsample_bytree': 0.6410965653177675
}

print("\n🟢 Processing train/test split...")

# =========================================================
# 3. 加载训练集 / 测试集数据
# =========================================================
X_train = np.load(os.path.join(data_dir, 'train_single.npy'))
y_train = np.load(os.path.join(data_dir, 'train_labels_single.npy'))

X_test  = np.load(os.path.join(data_dir, 'test_single.npy'))
y_test  = np.load(os.path.join(data_dir, 'test_labels_single.npy'))

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test  shape: {X_test.shape}")
print(f"y_test  shape: {y_test.shape}")

# =========================================================
# 4. 初始化并训练 XGBoost
# =========================================================
model = XGBRegressor(
    n_jobs=-1,
    verbosity=0,
    **best_params
)

model.fit(X_train, y_train)

# =========================================================
# 5. 测试集预测
# =========================================================
y_pred = model.predict(X_test)

# =========================================================
# 6. 保存预测值和真实值
# =========================================================
np.save(os.path.join(save_dir, 'single_test_y_pred.npy'), y_pred)
np.save(os.path.join(save_dir, 'single_test_y_true.npy'), y_test)

print("✅ Test split saved: single_test_y_pred.npy & single_test_y_true.npy")

# =========================================================
# 7. 保存模型
# =========================================================
model_path = os.path.join(save_dir, 'xgb_single_test.json')
model.save_model(model_path)
print(f"💾 XGBoost 模型已保存至: {model_path}")

print("\n🎉 XGBoost train/test split 预测结果已保存完毕！")


# 避免分母为 0
eps = 1e-12
y_test_safe = np.clip(y_test, eps, None)
y_pred_safe = np.clip(y_pred, eps, None)

# 计算 fold error：较大值 / 较小值
error = np.maximum(y_test_safe, y_pred_safe) / np.minimum(y_test_safe, y_pred_safe)

# 中值误差、平均误差
median_error = np.median(error)
mean_error = np.mean(error)

print("中值误差 (median absolute fold error):", median_error)
print("平均误差 (mean absolute fold error):", mean_error)

# 计算 5% 和 95% 分位数
lower_bound = np.percentile(error, 5)
upper_bound = np.percentile(error, 95)

# 截断异常值
error_clip = np.clip(error, lower_bound, upper_bound)

mean_error_clip = np.mean(error_clip)
median_error_clip = np.median(error_clip)

print("截断后平均误差:", mean_error_clip)
print("截断后中值误差:", median_error_clip)
