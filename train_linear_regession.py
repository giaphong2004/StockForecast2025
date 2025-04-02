#1.Import thư viện cần thiết
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

#2. Đọc dữ liệu
#gồm các cột : date, open(giá mở cửa), high(giá cao nhất), low(giá thấp nhất), close(giá cuối cùng trong ngày), volume(số lượng gd), Name(tên cổ phiếu)
data = pd.read_csv('dataset/spy.csv',index_col='Date')
data.head()

#3.Kiểm tra thông tin dữ liệu
print("Thông tin dữ liệu:")
print(data.info())

print("\nSố lượng giá trị thiếu:")
print(data.isnull().sum())


# 4. Chọn đặc trưng (X) và biến mục tiêu (y)

#Chọn close làm biến mục tiêu vì đây là giá cuối ngày là mức giá ổn định nhất khi tất không giao động như giá mở cửa open,giá cao nhất high, giá thấp nhất low,nó phản ánh xu hướng thị trường sau 1 ngày 
# 📌 Vì sao chọn dự đoán giá đóng cửa ?
#1️⃣ Giá mở cửa: Có thể bị ảnh hưởng bởi tin tức ngoài giờ hoặc tâm lý đầu cơ đầu phiên, chưa phản ánh đúng xu hướng ngày.
#2️⃣ Giá cao nhất:  & Giá thấp nhất (98): Chỉ là mức dao động, có thể bị tác động bởi các lệnh mua/bán lớn trong thời gian ngắn.
#3️⃣ Giá đóng cửa: Phản ánh chính xác nhất giá trị cuối cùng của cổ phiếu sau một ngày giao dịch, khi cung cầu đã ổn định.

X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']

# 5. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Xây dựng mô hình hồi quy tuyến tính(tìm w )
model = LinearRegression()
model.fit(X_train, y_train)
print("Hệ số hồi quy: ", model.coef_)

#7. Dự đoán trên tập kểm tra
y_pred = model.predict(X_test)

#8.Hàm dự đoán
def predict_price(open_price, high_price, low_price, volume):
    input_data = np.array([[open_price, high_price, low_price, volume]])
    prediction = model.predict(input_data)[0]
    return prediction

# 9. Đánh giá mô hình
if __name__ == "__main__":
    print("Độ lệch trung bình tuyệt đối(MAE): ", mean_absolute_error(y_test, y_pred))
    print("Hệ số xác định(R²): ", r2_score(y_test, y_pred))

# 10. Vẽ biểu đồ
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, label='Giá trị dự đoán so với thực tế', alpha=0.6)


    # Tự động đặt giới hạn trục
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val,max_val], [min_val,max_val], color='red', linestyle='--')

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.title("Biểu đồ dự đoán giá cổ phiếu(1993-2025)")
    plt.xlabel("Giá cuối ngày trong tập kiểm tra")
    plt.ylabel("Giá dự đoán")
    plt.legend()
    plt.show()







