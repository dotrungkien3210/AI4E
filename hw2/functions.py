import numpy as np
# tạo theo công thức h(x) = 0(0) + [0(1)*x1] + [0(2)*x2]+.......+[0(n)*xn
#nhân vô hướng ma trận X với vecto theta
def predict(X,theta):
    return X @ theta
def computeCost(X,y,theta):
    #gọi hàm predict bên trên và thực hiện
    predicted = predict(X,theta)
    # bình phương (giá trị giả thiết trừ giá trị thực tế): [h(x)-y]^2
    sqr_error = (predicted - y)**2
    # cộng tổng tất cả các giá trị bên trên lại
    sum_error = np.sum(sqr_error)
    # lấy số lượng mẫu thử
    m = np.size(y)
    #thực hiện phép tính cuối cùng
    J = (1/(2*m))*sum_error
    # trả lại kết quả
    return J
# cách 2 đơn giản hơn
def computeCost_vec(X,y,theta):
    error = predict(X,theta) - y
    m = np.size(y)
    J = (1/(2*m))*np.transpose(error)@error
    return J
#alpha là hệ số ta đặt cho là 0.02 còn iter là số lần lặp tối đa 5000
def GradientDescent(X,y,alpha=0.02,iter=5000):
    # hàm zeros thể hiện giá trị ban đầu của theta = 0
    # hàm theta là một hàng lần lượt các giá trị hệ số theta của X
    theta = np.zeros(np.size(X, 1))
    #array lưu giá trị J trong quá trình lặp ()
    # kích thước là iter*2, cột đầu chỉ là các số từ 1 đến iter để tiện cho việc plot.
    # cột 2 là các giá trị inter tương ứng
    # Kích thước được truyền vào qua một tupple
    J_hist = np.zeros((iter,2))
    # kích thước của training set
    m = np.size(y)
    # ma trận ngược (đảo hàng và cột) của X
    X_T = np.transpose(X)
    # biến tạm để kiểm tra tiến độ Gradient Descent
    # sau khi thực hiện phép tính giá trị J sẽ giảm từ từ và biến này in ra để quan sát
    pre_cost = computeCost(X, y, theta)
    # mỗi vòng lặp sẽ
    for i in range(0, iter):
        # tính sai số (predict – y)
        error = predict(X, theta) - y
        # thực hiện gradient descent để thay đổi theta
        theta = theta - (alpha / m) * (X_T @ error)
        # tính J hiện tại
        cost = computeCost(X, y, theta)
        # so sánh với J của vòng lặp trước để kiểm tra chênh lệch
        if np.round(cost, 15) == np.round(pre_cost, 15):
            # in ra vòng lặp hiện tại và J để dễ debug
            print('Reach optima at I = % d; J = % .6f' % (i, cost))
            # --bên trong câu điều kiện kiểm tra cost == pre_cost—
            # thêm tất cả các index còn lại sau khi break
            J_hist[i:, 0] = range(i, iter)
            # giá trị J sau khi break sẽ như cũ
            J_hist[i:, 1] = cost
            # thoát vòng lặp
            break
        # cập nhật pre_cost
        pre_cost = cost
        J_hist[i,0] = i
        J_hist[i,1] = cost
    yield theta
    yield J_hist
