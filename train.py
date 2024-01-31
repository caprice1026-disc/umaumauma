import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime

def train_and_save_model(train_data, model_index):
    # 現在の日時を取得
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")

    # モデル名を生成
    model_name = f"trained_model_{model_index}_{current_datetime}"

    # ニューラルネットワークモデルの定義
    model = Sequential()
    model.add(Dense(24, input_dim=train_data['X_train'].shape[1], activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))

    # モデルのコンパイル
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # モデルの訓練（早期停止を適用）
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(train_data['X_train'], train_data['y_train'], epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # モデルの評価
    loss, accuracy = model.evaluate(train_data['X_test'], train_data['y_test'])
    print(f"{model_name} - Loss: {loss}, Accuracy: {accuracy}")

    # モデルをGCSに保存（Keras形式で保存）
    model.save(f"{model_name}.keras")
    blob = bucket.blob(f"{model_name}.keras")
    blob.upload_from_filename(f"{model_name}.keras")

# Google Cloud Storageからデータをダウンロード
client = storage.Client()
bucket = client.get_bucket('datastrage')

# 学習データ
blob = bucket.blob('train_data.csv')
blob.download_to_filename('train_data.csv')
train_df = pd.read_csv('train_data.csv')
X_train = train_df.drop(['target_column'], axis=1)
y_train = train_df['target_column']
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# テストデータ
blob = bucket.blob('test_data.csv')
blob.download_to_filename('test_data.csv')
test_df = pd.read_csv('test_data.csv')
X_test = test_df.drop(['target_column'], axis=1)
y_test = test_df['target_column']
X_test = scaler.transform(X_test)

train_data = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}

# 複数のモデルを訓練
for i in range(1, 4):
    train_and_save_model(train_data, i)
