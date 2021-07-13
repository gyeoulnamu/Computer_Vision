from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt
from scipy import signal as sg
import numpy as np
import cv2
import os

# === laws texture 계산 함수
def laws_texture(gray_image):
    (rows, cols) = gray_image.shape[:2]  #이미지의 길이(rows)와 너비(cols)

    # === 이미지 전처리 ===
    smooth_kernel = (1/25)*np.ones((5,5))  # smoothing filter 만들기
    gray_smooth = sg.convolve(gray_image, smooth_kernel, "same")  # 흑백이미지 smoothing하기
    gray_processed = np.abs(gray_image - gray_smooth)  # 원본이미지에서 smoothing된 이미지 빼기

    # === Law's Texture filter ===
    filter_vectors = np.array([[1, 4, 6, 4, 1], [-1, -2, 0, 2, 1], [-1, 0, 2, 0, 1], [1, -4, 6, -4, 1]])  # L5, E5, S5, R5
                                                                                                          # 0:L5L5, 1:L5E5, 2:L5S5, 3:L5R5
                                                                                                          # 4:E5L5, 5:E5E5, 6:E5S5, 7:E5R5
                                                                                                          # 8:S5L5, 9:S5E5, 10:S5S5, 11:S5R5
                                                                                                          # 12:R5L5, 13:R5E5, 14:R5S5, 15:R5R5
    filters = []  # 16(4x4)개 filter를 저장할 filters
    for i in range(4):
        for j in range(4):
            filters.append(np.matmul(filter_vectors[i][:].reshape(5,1), filter_vectors[j][:].reshape(1,5)))  # 매트릭스 곱하기 연산을 통해 filter값 계산
            
    # === Convolution 연산 및 convmap 조합 ===
    conv_maps = np.zeros((rows, cols, 16))  # 계산된 convolution 결과를 저장할 conv_maps
    for i in range(len(filters)):
        conv_maps[:, :, i] = sg.convolve(gray_processed, filters[i], 'same')  # 전처리된 이미지에 16개 필터 적용
        
    # === 9+1개 중요한 texture map 계산 ===
    texture_maps = list()
    texture_maps.append((conv_maps[:, :, 1]+conv_maps[:, :, 4])//2)  # L5E5 / E5L5
    texture_maps.append((conv_maps[:, :, 2]+conv_maps[:, :, 8])//2)  # L5S5 / S5L5
    texture_maps.append((conv_maps[:, :, 3]+conv_maps[:, :, 12])//2)  # L5R5 / R5L5
    texture_maps.append((conv_maps[:, :, 7]+conv_maps[:, :, 13])//2)  # E5R5 / R5E5
    texture_maps.append((conv_maps[:, :, 6]+conv_maps[:, :, 9])//2)  # E5S5 / S5E5
    texture_maps.append((conv_maps[:, :, 11]+conv_maps[:, :, 14])//2)  # S5R5 / R5S5
    texture_maps.append(conv_maps[:, :, 10])  #S5S5
    texture_maps.append(conv_maps[:, :, 5])  #E5E5
    texture_maps.append(conv_maps[:, :, 15])  #R5R5
    texture_maps.append(conv_maps[:, :, 0])  #L5L5 (use to norm TEM)

    # === Law's texture energy 계산 ===
    TEM = list()
    for i in range(9):
        TEM.append(np.abs(texture_maps[i]).sum() / np.abs(texture_maps[9]).sum())  # TEM계산 및 L5L5 값으로 정규화
    return TEM  # 9차원의 TEM feature 추출: list

# === 이미지 패치에서 특징 추출 ===
train_dir = './archive/seg_train/seg_train'  # train data 경로
test_dir = './archive/seg_test/seg_test'  # test data 경로
classes = ['buildings', 'forest', 'mountain', 'sea']  # 클래스 이름

X_train = []  # train 데이터를 저장할 list
Y_train = []  # train 라벨을 저장할 list

PATCH_SIZE = 50  # 이미지 패치 사이즈
np.random.seed(1234)
for idx, texture_name in enumerate(classes):  # 각 class 마다
    image_dir = os.path.join(train_dir, texture_name)  # class image가 있는 경로
    for image_name in os.listdir(image_dir):  # 경로에 있는 모든 이미지에 대해
        image = cv2.imread(os.path.join(image_dir, image_name))  # 이미지 불러오기
        image_s = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)  # 이미지를 100x100으로 축소
        
        for _ in range(10):  # 이미지에서 random하게 10개 패치 자르기
            h = np.random.randint(100-PATCH_SIZE)  # 랜덤하게 자를 위치 선정
            w = np.random.randint(100-PATCH_SIZE)
            
            image_p = image_s[h:h+PATCH_SIZE, w:w+PATCH_SIZE]  # 이미지 패치 자르기
            image_p_gray = cv2.cvtColor(image_p, cv2.COLOR_BGR2GRAY)  # 이미지를 흑백으로 변환
            #image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 이미지를 HSV로 변환
            glcm = greycomatrix(image_p_gray, distances=[1], angles=[np.pi/2], levels=256, symmetric=False, normed=True)  # GLCM co-occurrence 계산
            X_train.append([greycoprops(glcm, 'contrast')[0, 0], greycoprops(glcm, 'correlation')[0, 0]] + laws_texture(image_p_gray))  # GLCM dissimilarity, correlation 특징 추가 (2차원) + laws texture 특징 추가 (9차원)
            Y_train.append(idx)  # 라벨 추가

X_train = np.array(X_train)  # list를 numpy array로 변경
Y_train = np.array(Y_train)
print('train data:  ', X_train.shape)
print('train label: ', Y_train.shape)

X_test = []  # test 데이터를 저장할 list
Y_test = []  # test 라벨을 저장할 list

for idx, texture_name in enumerate(classes):
    image_dir = os.path.join(test_dir, texture_name)
    for image_name in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, image_name))
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = greycomatrix(image_gray, distances = [1], angles=[np.pi/2], levels=256, symmetric=False, normed=True)
        X_test.append([greycoprops(glcm, 'contrast')[0, 0], greycoprops(glcm, 'correlation')[0, 0]] + laws_texture(image_gray))
        Y_test.append(idx)
        
X_test = np.array(X_test)
Y_test = np.array(Y_test)
print('test data:   ', X_test.shape)
print('test label:  ', Y_test.shape)

# === 신경망에 필요한 모듈 ===
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchsummary import summary

# === 데이터셋 클래스 ===
class textureDataset(Dataset):  # 데이터셋 클래스
    def __init__(self, features, labels):  # 초기화 함수
        self.features = features  # texture 특징
        self.labels = labels  # texture 라벨
        
    def __len__(self):  # 데이터셋 크기 반환
        return len(self.labels)
        
    def __getitem__(self, idx):  # idx번째 샘플을 반환
        if torch.is_tensor(idx):  # idx가 pytorch tensor면
            idx = idx.tolist()  # idx를 list로 변환
        feature = self.features[idx]
        label = self.labels[idx]
        sample = (feature, label)  # idx번째 특징과 라벨을 샘플로 묶어 반환
        
        return sample
        
# === 신경망 모델 클래스 ===
class MLP(nn.Module):  # MLP class
    def __init__(self, input_dim, hidden_dim, output_dim):  # 초기화 함수
        super(MLP, self).__init__()  # 기반 클래스 nn.Module을 초기화
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # input_dim x hidden_dim
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # hidden_dim x hidden_dim
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # hidden_dim x output_dim
        
    def forward(self, x):  # x: input_dim
        out = self.fc1(x)  # out: hidden_dim
        out = self.relu(out)  # out: hidden_dim
        out = self.fc2(out)  # out: hidden_dim
        out = self.relu(out)  # out: hidden_dim
        out = self.fc3(out)  # out: output_dim
        
        return out
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU: 'cuda', CPU: 'cpu'

batch_size = 10  # batch size
learning_rate=0.01  # 학습률
n_epoch = 500  # 전체 데이터셋을 반복학습 할 횟수

Train_data = textureDataset(features=X_train, labels=Y_train)  # 학습 데이터 정의
Test_data = textureDataset(features=X_test, labels=Y_test)  # 테스트 데이터 정의

Trainloader = DataLoader(Train_data, batch_size=batch_size, shuffle=True)  # 학습 데이터 로더 정의
Testloader = DataLoader(Test_data, batch_size=batch_size)  # 테스트 데이터 로더 정의

net = MLP(11, 8, 4)  # MLP 모델 정의
net.to(device)  # 모델을 device로 보내기
summary(net, (11,), device='cuda' if torch.cuda.is_available() else 'cpu')  # 모델 layer 출력

optimizer = optim.SGD(net.parameters(), lr=learning_rate)  # 옵티마이저 정의
criterion = nn.CrossEntropyLoss()  # loss 계산식 정의

train_losses = []  # 학습 loss를 저장할 list 정의
train_accs = []  # 학습 accuracy를 저장할 list 정의
test_losses = []  # 테스트 lost를 저장할 list 정의
test_accs = []  # 테스트 accuracy를 저장할 list 정의

# === 학습 ===
for epoch in range(n_epoch):  # 각 epoch마다
    train_loss = 0.0  # 학습 loss 0으로 초기화
    evaluation = []  # 예측 정확 여부 저장할 list
    net.train()  # 모델을 학습 모드로 전환
    for i, data in enumerate(Trainloader, 0):  # 각 batch마다
        features, labels = data  # 데이터를 특징과 라벨로 나누기
        labels = labels.long().to(device)  # 라벨을 long 형태로 변환 후 device로 보내기
        features = features.to(device)  # 특징을 device로 보내기
        optimizer.zero_grad()  # optimizer의 gradient를 0으로 초기화
        
        outputs = net(features.to(torch.float))  # 특징을 float형으로 변환 후 모델에 입력
        _, predicted = torch.max(outputs.cpu().data, 1)  # 출력의 제일 큰 값의 index 반환
        evaluation.append((predicted==labels.cpu()).tolist())  # 정답과 비교하여 True False 값을 저장
        loss = criterion(outputs, labels)  # 출력과 라벨을 비교하여 loss 계산
        
        loss.backward()  # 역전파, 기울기 계산
        optimizer.step()  # 가중치 값 업데이트, 학습 한번 진행
        
        train_loss += loss.item()  # loss를 train_loss에 누적
    train_loss = train_loss/(i+1)  # 평균 train_loss 구하기
    evaluation = [item for sublist in evaluation for item in sublist]  # [True, false] 값을 list로 저장
    train_acc = sum(evaluation)/len(evaluation)  # True인 비율 계산
    
    train_losses.append(train_loss)  # 해당 epoch의 train loss 기록
    train_accs.append(train_acc)  # 해당 epoch의 train acc 기록
    
    # === 테스트 ===
    if (epoch+1) % 1 == 0:
        test_loss = 0.0  # 테스트 loss 초기화
        evaluation = []
        net.eval()  # 모델을 평가 모드로 전환
        for i, data in enumerate(Testloader, 0):
            features, labels = data
            labels = labels.long().to(device)
            features = features.to(device)
            
            outputs = net(features.to(torch.float))
            _, predicted = torch.max(outputs.cpu().data, 1)
            evaluation.append((predicted==labels.cpu()).tolist())
            loss = criterion(outputs, labels)
            test_loss += loss.item()  # loss를 test_loss에 누적
        test_loss = test_loss/(i+1)  # 평균 test_loss 구하기
        evaluation = [item for sublist in evaluation for item in sublist]
        test_acc = sum(evaluation)/len(evaluation)
        
        test_losses.append(test_loss)  # 해당 epoch의 test loss 기록
        test_accs.append(test_acc)  # 해당 epoch의 test acc 기록
        
        print('[%d, %3d]\tloss: %.4f\tAccuracy : %.4f\t\tval-loss: %.4f\tval-Accuracy : %.4f' %(epoch+1, n_epoch, train_loss, train_acc, test_loss, test_acc))
        
# === 학습/테스트 loss/정확도 시각화 ===
plt.plot(range(len(train_losses)), train_losses, label='train loss')
plt.plot(range(len(test_losses)), test_losses, label='test loss')
plt.legend()
plt.show()
plt.plot(range(len(train_accs)), train_accs, label='train acc')
plt.plot(range(len(test_accs)), test_accs, label='test acc')
plt.legend()
plt.show()