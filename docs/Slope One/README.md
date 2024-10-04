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
\mathbf{\mu}_\mathbf{u}=\frac{\mathbf{1}}{\left|\mathbf{R}_\mathbf{u}\right|}\sum_{\mathbf{j}\in\mathbf{R}_\mathbf{u}}\mathbf{r}_{\mathbf{uj}}  
Trong đó:  
	: Là điểm trung bình mà người dùng *u* đã đánh giá.  
	: Là tập hợp các sản phẩm mà người dùng *u* đã đánh giá.  
  : Là số lượng sản phẩm người dùng *u* đã đánh giá.  
	: Là điểm đánh giá của người dùng *u* cho sản phẩm *j*.  
## 2. Công thức Độ Lệch
\mathbf{dev}\left(\mathbit{i},\mathbit{j}\right)=\frac{\mathbf{1}}{\left|\mathbf{U}_{\mathbit{ij}}\right|}\sum_{\mathbit{u}\in\mathbit{U}_{\mathbit{ij}}}\mathbit{r}_{\mathbit{ui}}-\mathbit{r}_{\mathbit{uj}}  

Trong đó:  
	: Là độ lệch trung bình giữa các điểm đánh giá của sản phẩm *i* và sản phẩm *j*.  
	: Là số lượng người dùng đã đánh giá cả hai sản phẩm *i* và *j*.  
	: Là tập hợp những người dùng đã đánh giá cả hai sản phẩm *i* và *j*.  
	: Là điểm đánh giá của người dùng *u* cho sản phẩm *i*.  
	: Là điểm đánh giá của người dùng *u* cho sản phẩm *j*.  
## 3. Công thức Dự đoán
\widehat{\mathbit{r}_{\mathbit{ui}}}=\mathbf{\mu}_\mathbit{u}+\frac{\mathbf{1}}{\left|\mathbit{R}_\mathbit{i}\left(\mathbit{u}\right)\right|}\sum_{\mathbit{j}\in\mathbit{R}_\mathbit{i}\left(\mathbit{u}\right)}\mathbit{dev}\left(\mathbit{i},\mathbit{j}\right)

Trong đó:  
	: Là điểm dự đoán mà người dùng *u* sẽ cho sản phẩm *i*.  
	: Là điểm trung bình mà người dùng *u* đã đánh giá. Nó phản ánh cách người dùng này thường đánh giá sản phẩm (ví dụ: một số người dùng thường cho điểm cao hơn những người khác).  
	: Là tập hợp các sản phẩm liên quan, tức là tập hợp các sản phẩm *j* mà người dùng u đã đánh giá và cũng có ít nhất một người dùng chung với sản phẩm *i*.  
	: Là số lượng sản phẩm liên quan trong .  
	: Là độ lệch trung bình giữa các điểm đánh giá của sản phẩm *i* và sản phẩm *j*.  
# III. Algorithm
**Bước 1: Khởi tạo dữ liệu**  
	Thu thập dữ liệu: Lấy dữ liệu từ các người dùng và các sản phẩm, tạo ra một ma trận đánh giá. Mỗi hàng trong ma trận đại diện cho một người dùng, và mỗi cột đại diện cho một sản phẩm. Giá trị trong ma trận là điểm đánh giá mà người dùng đã cho sản phẩm (hoặc có thể là một giá trị None/NaN nếu người dùng chưa đánh giá sản phẩm).  
**Bước 2: Tính toán điểm trung bình của người dùng**  
	Với mỗi người dùng *u*:   
\mathbf{\mu}_\mathbf{u}=\frac{\mathbf{1}}{\left|\mathbf{R}_\mathbf{u}\right|}\sum_{\mathbf{j}\in\mathbf{R}_\mathbf{u}}\mathbf{r}_{\mathbf{uj}}  
Trong đó \mathbf{R}_\mathbf{u} là tập hợp các sản phẩm mà người dùng *u* đã đánh giá.  

**Bước 3: Tính độ lệch giữa các sản phẩm**  
	Tạo một ma trận độ lệch \mathbf{dev}\left(\mathbit{i},\mathbit{j}\right)cho tất cả các cặp sản phẩm *i* và *j*:  
	Đối với mỗi cặp sản phẩm *i* và *j*:  
	Tìm tất cả người dùng *u* đã đánh giá cả hai sản phẩm *i* và *j*.  
	Tính độ lệch:  
\mathbf{dev}\left(\mathbit{i},\mathbit{j}\right)=\frac{\mathbf{1}}{\left|\mathbf{U}_{\mathbit{ij}}\right|}\sum_{\mathbit{u}\in\mathbit{U}_{\mathbit{ij}}}\mathbit{r}_{\mathbit{ui}}-\mathbit{r}_{\mathbit{uj}}  

**Bước 4: Dự đoán điểm đánh giá**  
	Để dự đoán điểm đánh giá cho người dùng *u* trên sản phẩm *i*:  
	Tính:  
\widehat{\mathbit{r}_{\mathbit{ui}}}=\mathbf{\mu}_\mathbit{u}+\frac{\mathbf{1}}{\left|\mathbit{R}_\mathbit{i}\left(\mathbit{u}\right)\right|}\sum_{\mathbit{j}\in\mathbit{R}_\mathbit{i}\left(\mathbit{u}\right)}\mathbit{dev}\left(\mathbit{i},\mathbit{j}\right)  
