slope one algorithm here
# I.	Ideas
Các thuật toán đề xuất thường được phân loại thành ba nhóm chính: **thuật toán dựa trên nội dung (Content-based Filtering)**, **thuật toán lọc cộng tác (Collaborative Filtering)** và **thuật toán lai (Hybrid Recommendation Algorithm)**. Trong đó, **thuật toán lọc cộng tác (Collaborative Filtering)** được coi là quan trọng nhất và thường được chia thành hai loại: **lọc cộng tác dựa trên người dùng (User-based Collaborative Filtering)** và **lọc cộng tác dựa trên sản phẩm (Item-based Collaborative Filtering)**.  
**Thuật toán lọc cộng tác (CF)** nhằm dự đoán sở thích hoặc hành vi của người dùng dựa trên thông tin từ những người dùng khác. Cơ chế hoạt động của thuật toán này dựa trên giả định rằng nếu hai người dùng có xu hướng thích những sản phẩm giống nhau trong quá khứ, thì khả năng cao họ cũng sẽ có sở thích tương tự trong tương lai.  
Ví dụ: Giả sử chúng ta muốn dự đoán liệu một người dùng có thích album mới của Celine Dion hay không, dựa trên thông tin rằng anh ta đã cho điểm 5/5 cho album của Beatles.  
Trong **Item-based CF**, thuật toán dự đoán điểm đánh giá của một sản phẩm dựa trên các đánh giá cho những sản phẩm tương tự. Một trong những kỹ thuật phổ biến được sử dụng là **hồi quy tuyến tính** (f(x) = ax + b). Tuy nhiên, nếu có 1.000 sản phẩm, sẽ có tới 1.000.000 hàm hồi quy tuyến tính cần phải được xây dựng, và như vậy, có thể cần đến 2.000.000 hệ số hồi quy. Đây chính là điểm yếu của phương pháp này, do việc xây dựng quá nhiều mô hình có thể dẫn đến vấn đề **quá khớp (Overfitting)**.  
Để giải quyết nhược điểm này, vào năm 2005, Daniel Lemire và Anna Maclachlan đã đề xuất một thuật toán đơn giản hơn có tên là **Slope One**. Thuật toán này dựa trên sự chênh lệch (difference) giữa các đánh giá của người dùng cho từng cặp sản phẩm. Cụ thể, nếu một người dùng thích sản phẩm A hơn sản phẩm B, thì sự chênh lệch giữa hai đánh giá có thể được sử dụng để dự đoán mức độ yêu thích của anh ta với một sản phẩm mới. Điều này có nghĩa là, sự khác biệt giữa điểm đánh giá của hai sản phẩm có thể được dùng để ước lượng điểm đánh giá cho một sản phẩm mà người dùng chưa đánh giá.  
Thuật toán **Slope One** là dạng đơn giản nhất của **Item-based CF**. Tính đơn giản giúp cho việc triển khai dễ dàng và hiệu quả hơn, trong khi độ chính xác của nó thường tương đương với các thuật toán phức tạp và tốn kém hơn về mặt tính toán.  
# II.	Formulation
Cho một ma trận đánh giá *R* với kích thước *m × n*, trong đó:  
•	*m* là số lượng người dùng.  
•	*n* là số lượng sản phẩm.  
•	Mỗi phần tử **R*ij*** trong ma trận đại diện cho điểm đánh giá mà người dùng *i* đưa ra cho sản phẩm *j*.  
•	Nếu **R*ij*** = *∅*, điều đó có nghĩa là người dùng *i* chưa đánh giá sản phẩm *j*.  
Mục tiêu: Dự đoán điểm đánh giá của người dùng đối với các sản phẩm mà họ chưa đánh giá (các ô **R*ij*** có giá trị rỗng trong ma trận) dựa trên các đánh giá hiện có.  
## 1. Tính điểm trung bình của người dùng
![](../images/calculateAverage.png)  
Trong đó:  
•![](../images/uu.png) : Là điểm trung bình mà người dùng *u* đã đánh giá.  
•***R<sub>u</sub>***: Là tập hợp các sản phẩm mà người dùng *u* đã đánh giá.  
•***R<sub>u</sub>***: Là số lượng sản phẩm người dùng *u* đã đánh giá.  
•***r<sub>uj</sub>***: Là điểm đánh giá của người dùng *u* cho sản phẩm *j*.  
## 2. Công thức Độ Lệch 
![](../images/itemDeviation.png)  
Trong đó:  
•***dev(i,j)***: Là độ lệch trung bình giữa các điểm đánh giá của sản phẩm *i* và sản phẩm *j*.  
•***|U<sub>ij</sub>|***: Là số lượng người dùng đã đánh giá cả hai sản phẩm *i* và *j*.  
•***U<sub>ij</sub>***: Là tập hợp những người dùng đã đánh giá cả hai sản phẩm *i* và *j*.  
•***r<sub>ui</sub>***: Là điểm đánh giá của người dùng *u* cho sản phẩm *i*.  
•***r<sub>uj</sub>***: Là điểm đánh giá của người dùng *u* cho sản phẩm *j*.  
## 3. Công thức Dự đoán
![](../images/predictedRating.png)  
Trong đó:  
•![](../images/rui.png) : Là điểm dự đoán mà người dùng *u* sẽ cho sản phẩm *i*.  
•![](../images/uu.png) : Là điểm trung bình mà người dùng *u* đã đánh giá. Nó phản ánh cách người dùng này thường đánh giá sản phẩm (ví dụ: một số người dùng thường cho điểm cao hơn những người khác).  
•***R<sub>i</sub>(u)***: Là tập hợp các sản phẩm liên quan, tức là tập hợp các sản phẩm *j* mà người dùng u đã đánh giá và cũng có ít nhất một người dùng chung với sản phẩm *i*.  
•|***R<sub>i</sub>(u)***|: Là số lượng sản phẩm liên quan trong ***Ri(u)***.  
•***dev(i,j)***: Là độ lệch trung bình giữa các điểm đánh giá của sản phẩm *i* và sản phẩm *j*.  
# III. Algorithm
**Bước 1: Khởi tạo dữ liệu**  
	Thu thập dữ liệu: Lấy dữ liệu từ các người dùng và các sản phẩm, tạo ra một ma trận đánh giá. Mỗi hàng trong ma trận đại diện cho một người dùng, và mỗi cột đại diện cho một sản phẩm. Giá trị trong ma trận là điểm đánh giá mà người dùng đã cho sản phẩm (hoặc có thể là một giá trị None/NaN nếu người dùng chưa đánh giá sản phẩm).  
**Bước 2: Tính toán điểm trung bình của người dùng**  
	Với mỗi người dùng *u*:   
	![](../images/calculateAverage.png)   
	Trong đó ***Ru*** là tập hợp các sản phẩm mà người dùng *u* đã đánh giá.  

**Bước 3: Tính độ lệch giữa các sản phẩm**  
	Tạo một ma trận độ lệch ***dev(i,j)*** cho tất cả các cặp sản phẩm *i* và *j*:  
	Đối với mỗi cặp sản phẩm *i* và *j*:  
	Tìm tất cả người dùng *u* đã đánh giá cả hai sản phẩm *i* và *j*.  
	Tính độ lệch:  
	![](../images/itemDeviation.png) 

**Bước 4: Dự đoán điểm đánh giá**  
	Để dự đoán điểm đánh giá cho người dùng *u* trên sản phẩm *i*:  
	Tính:  
	![](../images/predictedRating.png) 
## Mã Giả (Pseudocode)
```Mã Giả (Pseudocode)
function slopeOnePredict(userId, itemId, ratings):
    // Step 1: Initialize data
    userAvgRating = {} // Dictionary to store average rating of each user  
    itemDeviation = {} // Dictionary to store deviations between items  
  
    // Step 2: Calculate user averages  
    for user in ratings:  
        userRatings = ratings[user] // Get ratings for the user  
        userAvgRating[user] = calculateAverage(userRatings)  
  
    // Step 3: Calculate item deviations  
    for itemA in ratings.items():  
        for itemB in ratings.items():  
            if itemA != itemB:  
                commonUsers = findCommonUsers(itemA, itemB, ratings)  
                if commonUsers:  
                    deviation = calculateDeviation(itemA, itemB, commonUsers, ratings)  
                    itemDeviation[(itemA, itemB)] = deviation  
  
    // Step 4: Predict the rating  
    relevantItems = findRelevantItems(userId, ratings)  
    predictedRating = userAvgRating[userId]  
      
    if relevantItems:  
        for item in relevantItems:  
            predictedRating += (1 / len(relevantItems)) * itemDeviation[(itemId, item)]  
      
    return predictedRating  

function calculateAverage(userRatings):  
    total = 0  
    count = 0  
    for rating in userRatings:  
        total += rating  
        count += 1  
    return total / count if count > 0 else 0  
  
function calculateDeviation(itemA, itemB, commonUsers, ratings):  
    totalDeviation = 0  
    for user in commonUsers:  
        totalDeviation += (ratings[user][itemA] - ratings[user][itemB])  
    return totalDeviation / len(commonUsers)  
  
function findCommonUsers(itemA, itemB, ratings):  
    commonUsers = []  
    for user in ratings:  
        if itemA in ratings[user] and itemB in ratings[user]:  
            commonUsers.append(user)  
    return commonUsers  
  
function findRelevantItems(userId, ratings):  
    // Return the list of items rated by the user  
    return ratings[userId].keys()
```

## IV.	Evaluation
**Ưu điểm:** 
**Đơn giản và dễ hiểu:** Slope One có cấu trúc toán học đơn giản, dễ hiểu và dễ triển khai. Việc tính toán trung bình và độ lệch giữa các sản phẩm không quá phức tạp.  
**Hiệu quả:** Slope One thường cho ra kết quả chính xác và hiệu quả trong việc dự đoán điểm đánh giá, đặc biệt là khi có nhiều người dùng đã đánh giá các sản phẩm tương tự.  
**Khả năng mở rộng:** Thuật toán có khả năng mở rộng tốt khi số lượng sản phẩm và người dùng tăng lên. Nó vẫn hoạt động hiệu quả ngay cả khi có nhiều sản phẩm và nhiều đánh giá.  
**Giảm thiểu Overfitting:** So với một số phương pháp khác (như hồi quy tuyến tính phức tạp), Slope One có khả năng giảm thiểu hiện tượng overfitting do tính chất đơn giản của nó.  
**Ít yêu cầu về bộ nhớ:** Slope One cần ít bộ nhớ để lưu trữ các thông tin cần thiết cho việc dự đoán so với một số thuật toán phức tạp hơn.  
**Nhược điểm:**   
**Phụ thuộc vào dữ liệu đánh giá:** Thuật toán có thể hoạt động kém nếu không có đủ dữ liệu đánh giá hoặc nếu dữ liệu đánh giá không đầy đủ. Nếu người dùng chỉ đánh giá một số ít sản phẩm, độ chính xác của dự đoán sẽ giảm.  
**Chỉ áp dụng cho cặp sản phẩm:** Slope One chủ yếu dựa vào các cặp sản phẩm để tính toán độ lệch, điều này có thể hạn chế khả năng dự đoán khi không có nhiều sản phẩm liên quan giữa các người dùng.  
**Không xử lý tốt cho dữ liệu thưa thớt:** Khi dữ liệu rất thưa thớt (nhiều sản phẩm nhưng ít người dùng đánh giá), Slope One có thể gặp khó khăn trong việc tìm ra mối quan hệ giữa các sản phẩm và người dùng.  
**Giới hạn về mối quan hệ người dùng:** Slope One tập trung vào mối quan hệ giữa các sản phẩm hơn là giữa các người dùng. Nếu mối quan hệ giữa người dùng là quan trọng hơn, thuật toán có thể không hoạt động tốt.  
**Khả năng gợi ý giới hạn:** Do việc dự đoán phụ thuộc vào sự tương đồng giữa các sản phẩm đã có, Slope One có thể không gợi ý được các sản phẩm mới hoặc không phổ biến mà không có đánh giá từ người dùng.  

