import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 하이퍼파라미터 및 설정
HIDDEN_DIM = 512
NUM_TEXT_EXPERTS = 4
NUM_IMAGE_EXPERTS = 4
EXPERT_CAPACITY = 0.25
DEPTH_CAPACITY = 0.5
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
SEQ_LENGTH = 1024
NUM_EPOCHS = 10
VOCAB_SIZE = 1000
NUM_LAYERS = 4

class FFNBlock(nn.Module):
    """Feed-forward network block with SwiGLU activation"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SwiGLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, x):
        return self.ffn(x)

class MoMaLayer(nn.Module):
    """Mixture of Modality-aware Experts (MoMa) layer"""
    def __init__(self, hidden_dim, num_text_experts, num_image_experts, expert_capacity, depth_capacity):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_text_experts = num_text_experts
        self.num_image_experts = num_image_experts
        self.expert_capacity = expert_capacity
        self.depth_capacity = depth_capacity
        
        # 텍스트와 이미지 전문가 초기화
        self.text_experts = nn.ModuleList([FFNBlock(hidden_dim) for _ in range(num_text_experts)])
        self.image_experts = nn.ModuleList([FFNBlock(hidden_dim) for _ in range(num_image_experts)])
        
        # 메인 라우터
        self.text_router = nn.Linear(hidden_dim, num_text_experts)
        self.image_router = nn.Linear(hidden_dim, num_image_experts)
        self.mod_router = nn.Linear(hidden_dim, 1)
        
        # 보조 라우터
        self.aux_text_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_text_experts)
        )
        self.aux_image_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_image_experts)
        )
        self.aux_mod_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, modality_mask, use_aux_router=False):
        batch_size, seq_length, _ = x.shape
        
        # Mixture-of-Depths (MoD) 라우팅
        if use_aux_router:
            mod_scores = torch.sigmoid(self.aux_mod_router(x).squeeze(-1))
        else:
            mod_scores = torch.sigmoid(self.mod_router(x).squeeze(-1))
        _, mod_indices = torch.topk(mod_scores, k=int(self.depth_capacity * seq_length), dim=1)
        mod_mask = torch.zeros_like(modality_mask, dtype=torch.bool)
        mod_mask.scatter_(1, mod_indices, 1)
        
        # MoD 마스크 적용
        x = x[mod_mask]
        modality_mask = modality_mask[mod_mask]
        
        # 텍스트와 이미지 토큰 분리
        text_tokens = x[modality_mask == 1]
        image_tokens = x[modality_mask == 0]
        
        # 텍스트 전문가 라우팅
        if use_aux_router:
            text_scores = torch.sigmoid(self.aux_text_router(text_tokens))
        else:
            text_scores = self.text_router(text_tokens)
        text_top_k = int(self.expert_capacity * len(text_tokens))
        text_expert_outputs = []
        for i, expert in enumerate(self.text_experts):
            if use_aux_router:
                selected_indices = torch.where(text_scores[:, i] > 0.5)[0]
            else:
                _, selected_indices = torch.topk(text_scores[:, i], k=min(text_top_k, len(text_tokens)), dim=0)
            selected_tokens = text_tokens[selected_indices]
            expert_output = expert(selected_tokens)
            padded_output = torch.zeros_like(text_tokens)
            padded_output[selected_indices] = expert_output
            text_expert_outputs.append(padded_output)
        text_output = sum(text_expert_outputs)
        
        # 이미지 전문가 라우팅 (텍스트와 유사)
        if use_aux_router:
            image_scores = torch.sigmoid(self.aux_image_router(image_tokens))
        else:
            image_scores = self.image_router(image_tokens)
        image_top_k = int(self.expert_capacity * len(image_tokens))
        image_expert_outputs = []
        for i, expert in enumerate(self.image_experts):
            if use_aux_router:
                selected_indices = torch.where(image_scores[:, i] > 0.5)[0]
            else:
                _, selected_indices = torch.topk(image_scores[:, i], k=min(image_top_k, len(image_tokens)), dim=0)
            selected_tokens = image_tokens[selected_indices]
            expert_output = expert(selected_tokens)
            padded_output = torch.zeros_like(image_tokens)
            padded_output[selected_indices] = expert_output
            image_expert_outputs.append(padded_output)
        image_output = sum(image_expert_outputs)
        
        # 결과 합치기
        output = torch.zeros_like(x)
        output[modality_mask == 1] = text_output
        output[modality_mask == 0] = image_output
        
        # 원래 시퀀스 형태로 복원
        full_output = torch.zeros(batch_size, seq_length, self.hidden_dim, device=x.device)
        full_output[mod_mask] = output
        
        return full_output, mod_mask, text_scores, image_scores

class MoMaModel(nn.Module):
    """전체 MoMa 모델"""
    def __init__(self, vocab_size, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            MoMaLayer(hidden_dim, NUM_TEXT_EXPERTS, NUM_IMAGE_EXPERTS, EXPERT_CAPACITY, DEPTH_CAPACITY)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, modality_mask, use_aux_router=False):
        x = self.embedding(x)
        for layer in self.layers:
            x, _, _, _ = layer(x, modality_mask, use_aux_router)
        return self.output(x)

class DummyDataset(Dataset):
    """더미 데이터셋 (실제 데이터셋으로 교체 필요)"""
    def __init__(self, num_samples, seq_length, vocab_size):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length))
        self.modality_mask = torch.randint(0, 2, (num_samples, seq_length))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.modality_mask[idx]

def train_model(model, train_loader, num_epochs, learning_rate):
    """모델 학습 함수"""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch, (data, modality_mask) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data, modality_mask)
            loss = criterion(output.view(-1, output.size(-1)), data.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    return model

def train_aux_routers(model, train_loader, num_epochs, learning_rate):
    """보조 라우터 학습 함수"""
    aux_params = []
    for layer in model.layers:
        aux_params.extend(list(layer.aux_text_router.parameters()))
        aux_params.extend(list(layer.aux_image_router.parameters()))
        aux_params.extend(list(layer.aux_mod_router.parameters()))
    
    optimizer = optim.Adam(aux_params, lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        model.eval()
        total_loss = 0
        for batch, (data, modality_mask) in enumerate(train_loader):
            optimizer.zero_grad()
            with torch.no_grad():
                _, mod_masks, text_scores, image_scores = zip(*[layer(model.embedding(data), modality_mask) for layer in model.layers])
            
            aux_mod_output = torch.cat([layer.aux_mod_router(model.embedding(data)) for layer in model.layers], dim=0)
            aux_text_output = torch.cat([layer.aux_text_router(model.embedding(data)[modality_mask == 1]) for layer in model.layers], dim=0)
            aux_image_output = torch.cat([layer.aux_image_router(model.embedding(data)[modality_mask == 0]) for layer in model.layers], dim=0)
            
            mod_targets = torch.cat(mod_masks, dim=0).float()
            text_targets = torch.cat([s > 0 for s in text_scores], dim=0).float()
            image_targets = torch.cat([s > 0 for s in image_scores], dim=0).float()
            
            loss = criterion(aux_mod_output.squeeze(), mod_targets)
            loss += criterion(aux_text_output, text_targets)
            loss += criterion(aux_image_output, image_targets)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Aux Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    return model

def upcycle_model(model, train_loader, num_epochs, learning_rate):
    """모델 업사이클링 함수"""
    # 1단계: 1개의 전문가로 학습
    for layer in model.layers:
        layer.text_experts = nn.ModuleList([layer.text_experts[0]])
        layer.image_experts = nn.ModuleList([layer.image_experts[0]])
        layer.text_router = nn.Linear(HIDDEN_DIM, 1)
        layer.image_router = nn.Linear(HIDDEN_DIM, 1)
    
    model = train_model(model, train_loader, num_epochs // 2, learning_rate)
    
    # 2단계: 전문가 수 늘리기
    for layer in model.layers:
        layer.text_experts = nn.ModuleList([layer.text_experts[0]] * NUM_TEXT_EXPERTS)
        layer.image_experts = nn.ModuleList([layer.image_experts[0]] * NUM_IMAGE_EXPERTS)
        layer.text_router = nn.Linear(HIDDEN_DIM, NUM_TEXT_EXPERTS)
        layer.image_router = nn.Linear(HIDDEN_DIM, NUM_IMAGE_EXPERTS)
    
    model = train_model(model, train_loader, num_epochs // 2, learning_rate)
    
    return model

def main():
    # 데이터셋과 데이터 로더 생성
    train_dataset = DummyDataset(10000, SEQ_LENGTH, VOCAB_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 모델 생성
    model = MoMaModel(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)

    # 업사이클링 적용
    model = upcycle_model(model, train_loader, NUM_EPOCHS, LEARNING_RATE)

    # 보조 라우터 학습
    model = train_aux_routers(model, train_loader, NUM_EPOCHS // 2, LEARNING_RATE)

    # 추론 예시
    model.eval()
    with torch.no_grad():
        sample_data, sample_mask = next(iter(train_loader))
        output = model(sample_data, sample_mask, use_aux_router=True)
        print("Inference output shape:", output.shape)

if __name__ == "__main__":
    main()