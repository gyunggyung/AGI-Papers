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
VOCAB_SIZE = 65536  # 텍스트 토큰 + 이미지 코드북
IMAGE_CODEBOOK_SIZE = 8192
NUM_LAYERS = 4

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class FFNBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            SwiGLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, x):
        return self.ffn(x)

class MoMaLayer(nn.Module):
    def __init__(self, hidden_dim, num_text_experts, num_image_experts, expert_capacity, depth_capacity):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_text_experts = num_text_experts
        self.num_image_experts = num_image_experts
        self.expert_capacity = expert_capacity
        self.depth_capacity = depth_capacity
        
        self.text_experts = nn.ModuleList([FFNBlock(hidden_dim) for _ in range(num_text_experts)])
        self.image_experts = nn.ModuleList([FFNBlock(hidden_dim) for _ in range(num_image_experts)])
        
        self.text_router = nn.Linear(hidden_dim, num_text_experts)
        self.image_router = nn.Linear(hidden_dim, num_image_experts)
        self.mod_router = nn.Linear(hidden_dim, 1)
        
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
        mod_scores = torch.sigmoid(self.aux_mod_router(x) if use_aux_router else self.mod_router(x)).squeeze(-1)
        _, mod_indices = torch.topk(mod_scores, k=int(self.depth_capacity * seq_length), dim=1)
        mod_mask = torch.zeros_like(modality_mask, dtype=torch.bool).scatter_(1, mod_indices, 1)
        
        x = x[mod_mask]
        modality_mask = modality_mask[mod_mask]
        
        text_tokens = x[modality_mask == 1]
        image_tokens = x[modality_mask == 0]
        
        def route_and_process(tokens, router, aux_router, experts):
            scores = torch.sigmoid(aux_router(tokens) if use_aux_router else router(tokens))
            top_k = int(self.expert_capacity * len(tokens))
            outputs = []
            for i, expert in enumerate(experts):
                if use_aux_router:
                    indices = torch.where(scores[:, i] > 0.5)[0]
                else:
                    _, indices = torch.topk(scores[:, i], k=min(top_k, len(tokens)), dim=0)
                selected_tokens = tokens[indices]
                expert_output = expert(selected_tokens)
                padded_output = torch.zeros_like(tokens)
                padded_output[indices] = expert_output
                outputs.append(padded_output)
            return sum(outputs), scores
        
        text_output, text_scores = route_and_process(text_tokens, self.text_router, self.aux_text_router, self.text_experts)
        image_output, image_scores = route_and_process(image_tokens, self.image_router, self.aux_image_router, self.image_experts)
        
        output = torch.zeros_like(x)
        output[modality_mask == 1] = text_output
        output[modality_mask == 0] = image_output
        
        full_output = torch.zeros(batch_size, seq_length, self.hidden_dim, device=x.device)
        full_output[mod_mask] = output
        
        return full_output, mod_mask, text_scores, image_scores

class MoMaModel(nn.Module):
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

class ChameleonDataset(Dataset):
    def __init__(self, num_samples, seq_length, vocab_size, image_codebook_size):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length))
        self.modality_mask = torch.zeros((num_samples, seq_length), dtype=torch.bool)
        for i in range(num_samples):
            image_length = torch.randint(256, 1025, (1,)).item()  # 256~1024 범위의 이미지 토큰
            image_start = torch.randint(0, seq_length - image_length, (1,)).item()
            self.data[i, image_start:image_start+image_length] = torch.randint(0, image_codebook_size, (image_length,))
            self.modality_mask[i, image_start:image_start+image_length] = True
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.modality_mask[idx]

def train_model(model, train_loader, num_epochs, learning_rate):
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
    aux_params = [p for layer in model.layers for router in [layer.aux_text_router, layer.aux_image_router, layer.aux_mod_router] for p in router.parameters()]
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
            aux_text_output = torch.cat([layer.aux_text_router(model.embedding(data)[modality_mask]) for layer in model.layers], dim=0)
            aux_image_output = torch.cat([layer.aux_image_router(model.embedding(data)[~modality_mask]) for layer in model.layers], dim=0)
            
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
    train_dataset = ChameleonDataset(10000, SEQ_LENGTH, VOCAB_SIZE, IMAGE_CODEBOOK_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MoMaModel(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    model = upcycle_model(model, train_loader, NUM_EPOCHS, LEARNING_RATE)
    model = train_aux_routers(model, train_loader, NUM_EPOCHS // 2, LEARNING_RATE)

    model.eval()
    with torch.no_grad():
        sample_data, sample_mask = next(iter(train_loader))
        output = model(sample_data, sample_mask, use_aux_router=True)
        print("Inference output shape:", output.shape)

if __name__ == "__main__":
    main()

```
이 개선된 버전에서는 다음과 같은 변경사항이 있습니다:

1. Chameleon 모델의 특성을 반영한 `ChameleonDataset` 클래스를 추가했습니다. 이 클래스는 텍스트와 이미지 토큰이 섞인 시퀀스를 생성합니다.

2. SwiGLU 활성화 함수를 별도의 클래스로 구현하여 FFNBlock에 적용했습니다.

3. MoMaLayer의 forward 메서드를 더 간결하고 효율적으로 리팩토링했습니다.

4. vocab_size와 image_codebook_size를 논문에서 언급된 값으로 업데이트했습니다.

5. 라우팅 로직을 더 명확하게 구현하여 expert-choice 라우팅의 특성을 잘 반영했습니다.

이 코드는 논문에서 제안한 MoMa 아키텍처의 핵심 아이디어를 더 정확하게 구현하고 있습니다. 하지만 실제 대규모 학습을 위해서는 분산 학습, 메모리 최적화, 그리고 실제 데이터셋을 사용한 추가적인 구현이 필요할 것입니다.
```