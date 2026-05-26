import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state  # SSM의 상태 차원 (N)
        self.d_conv = d_conv    # 1D 로컬 컨볼루션 커널 크기
        self.expand = expand    # 차원 확장 계수 (E=2)
        self.d_inner = int(self.expand * self.d_model) # 확장된 차원

        # 1. 입력 듀얼 패스 분기 (Linear Projections)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        # 2. 메인 경로의 1D 로컬 컨볼루션
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=self.d_conv,
            groups=self.d_inner, # 각 채널별 독립 연산 (Depthwise)
            padding=self.d_conv - 1,
        )

        # 3. 선택적 SSM(Selective SSM)을 위한 입력 주도 가중치 프로젝션 레이어
        # 입력 x에 따라 실시간으로 변화하는 \Delta, B, C 생성
        self.x_proj = nn.Linear(self.d_inner, 1 + self.d_state * 2, bias=False)
        
        # 기본 고정 파라미터 A (S4D-Real 기반 음수 초기화 가정)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1).float().repeat(self.d_inner, 1)))
        
        # 4. 최종 출력 프로젝션
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        # x shape: (B, L, D) [Batch, Sequence_Length, d_model]
        B, L, _ = x.shape

        # Step 1: 듀얼 패스 분기 및 메인 경로 연산 준비
        # 입력 가중치 통과 후 메인 경로(x)와 게이팅 경로(res)로 분할
        projected = self.in_proj(x) # (B, L, 2 * d_inner)
        x_inner, res = projected.chunk(2, dim=-1) # 각각 (B, L, d_inner)

        # Step 2: 1D 로컬 컨볼루션 & SiLU 활성화 함수
        # Conv1d를 위해 차원 축 교환 (B, L, D) -> (B, D, L)
        x_inner = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_inner)[:, :, :L] # 패딩 잘라내기
        x_conv = x_conv.transpose(1, 2) # 원래 차원으로 복귀 (B, L, d_inner)
        x_act = F.silu(x_conv) # SiLU / Swish 활성화

        # Step 3: 선택적 SSM (Selective SSM) 연산
        y_ssm = self.selective_ssm(x_act)

        # Step 4: 게이트 합성 (Element-wise 곱)
        # 메인 경로의 결과와 처음에 갈라졌던 게이팅 경로(res)를 결합
        gated_output = y_ssm * F.silu(res)

        # Step 5: 최종 출력 투사
        output = self.out_proj(gated_output)
        return output

    def selective_ssm(self, x_act):
        """
        논문 3.2절의 알고리즘 2(S6) 파트 핵심 요약 구현
        (실제 논문에서는 성능을 위해 이 파트를 GPU SRAM 전용 커스텀 Cuda 커널로 병렬 처리함)
        """
        B, L, D = x_act.shape
        A = -torch.exp(self.A_log) # (D, N) 형태의 음수 행렬 보장

        # 입력 x_act에 의존하는 가변 파라미터 \Delta, B, C 추출
        x_proj_out = self.x_proj(x_act) # (B, L, 1 + 2*N)
        delta, B_t, C_t = torch.split(x_proj_out, [1, self.d_state, self.d_state], dim=-1)
        
        # \Delta는 가독성을 위해 softplus 통과 (논문 규칙)
        delta = F.softplus(delta) # (B, L, 1)

        # 출력 및 잠재 상태 변수 초기화
        y = torch.zeros_like(x_act)
        h = torch.zeros(B, D, self.d_state, device=x_act.device) # Hidden State (B, D, N)

        # 시퀀스 길이(L)를 따라 순환 연산 (RNN 방식)
        # *주의*: 논문은 이 루프를 병렬 스캔(Parallel Scan) 알고리즘으로 가속화함
        for t in range(L):
            x_t = x_act[:, t, :] # (B, D)
            delta_t = delta[:, t, :] # (B, 1)
            B_mat = B_t[:, t, :] # (B, N)
            C_mat = C_t[:, t, :] # (B, N)

            # 이산화 (Discretization) 공식 간단 적용
            # dA = exp(\Delta * A)
            # dB = \Delta * B
            dA = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0)) # (B, D, N)
            dB = delta_t.unsqueeze(-1) * B_mat.unsqueeze(1)       # (B, D, N)

            # 숨겨진 상태 업데이트: h_t = dA * h_{t-1} + dB * x_t
            h = dA * h + dB * x_t.unsqueeze(-1) # (B, D, N)

            # 출력 계산: y_t = C * h_t
            # 각 채널별로 잠재 상태와 C 행렬을 곱함
            y[:, t, :] = torch.sum(h * C_mat.unsqueeze(1), dim=-1) # (B, D)

        return y

# --- 모델 작동 테스트 ---
if __name__ == "__main__":
    # 임의의 입력 데이터 정의: BatchSize=2, SequenceLength=10, EmbeddingDim=64
    batch_size = 2
    seq_len = 10
    d_model = 64
    
    mamba_layer = MambaBlock(d_model=d_model)
    input_tensor = torch.randn(batch_size, seq_len, d_model)
    
    output_tensor = mamba_layer(input_tensor)
    
    print("입력 텐서 구조:", input_tensor.shape)
    print("출력 텐서 구조:", output_tensor.shape) # 입력 구조(B, L, D)가 그대로 유지됨