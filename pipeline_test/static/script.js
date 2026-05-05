document.addEventListener('DOMContentLoaded', () => {
    const plateOutInput = document.getElementById('plateOut');
    const plateInInput = document.getElementById('plateIn');
    const statusBadge = document.getElementById('statusBadge');
    
    // Hình ảnh khu vực Cắt biển (Dưới cùng)
    const imgCamOut = document.getElementById('imgCamOut'); 
    const imgCamIn = document.getElementById('imgCamIn');
    
    // Hình ảnh khu vực LPR (Chụp full màn hình lúc quẹt)
    const liveCamOut = document.getElementById('liveCamOut'); // CH1
    const mainCamIn = document.getElementById('mainCamIn');   // CH3
    
    const recentExitImg = document.getElementById('recentExitImg');

    // Text fields
    const timeInDisplay = document.getElementById('timeInDisplay');
    const timeOutDisplay = document.getElementById('timeOutDisplay');
    const priceDisplay = document.getElementById('priceDisplay');
    const durationDisplay = document.getElementById('durationDisplay');
    
    const ticketCodeDisplay = document.getElementById('ticketCodeDisplay');
    const customerTypeDisplay = document.getElementById('customerTypeDisplay');

    // Hàm so sánh 2 biển số AI đọc được (Lúc vào vs Lúc ra)
    function comparePlates() {
        if (!plateOutInput || !plateInInput || !statusBadge) return;
        const valOut = plateOutInput.value.toUpperCase().trim();
        const valIn = plateInInput.value.toUpperCase().trim();

        if (valOut !== valIn) {
            plateOutInput.classList.remove('border-slate-600', 'focus:border-blue-500', 'bg-slate-800');
            plateOutInput.classList.add('border-red-500', 'text-red-400', 'bg-red-900/20', 'focus:border-red-400');
            statusBadge.innerHTML = '❌ BIỂN SỐ KHÔNG KHỚP';
            statusBadge.className = 'bg-red-600/20 border border-red-500 text-red-400 font-bold text-center py-3 rounded-lg text-xl mb-6 shadow-[0_0_15px_rgba(239,68,68,0.1)] transition-colors duration-300';
        } else {
            plateOutInput.classList.add('border-slate-600', 'focus:border-blue-500', 'bg-slate-800');
            plateOutInput.classList.remove('border-red-500', 'text-red-400', 'bg-red-900/20', 'focus:border-red-400');
            statusBadge.innerHTML = '✅ BIỂN SỐ GIỐNG';
            statusBadge.className = 'bg-emerald-600/20 border border-emerald-500 text-emerald-400 font-bold text-center py-3 rounded-lg text-xl mb-6 shadow-[0_0_15px_rgba(16,185,129,0.1)] transition-colors duration-300';
        }
    }

    // Lắng nghe sự kiện nếu nhân viên bảo vệ tự sửa biển số bằng tay
    if (plateOutInput) {
        plateOutInput.addEventListener('input', comparePlates);
    }

    // Kết nối WebSocket với Backend
    const ws = new WebSocket("ws://localhost:8000/ws"); 

    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        console.log("Nhận dữ liệu:", data);

        // Hiển thị mã thẻ RFID gốc đầy đủ
        if(ticketCodeDisplay) ticketCodeDisplay.innerText = data.rfid;
        
        // Cập nhật text và màu cho Loại Khách Hàng (Tên, SĐT, Loại)
        if(customerTypeDisplay) {
            customerTypeDisplay.innerHTML = data.customer_type; // <-- ĐỔI THÀNH innerHTML
            
            if (data.customer_type.includes("Khách Đăng Ký")) {
                // Thêm inline-block và text-center để chữ xuống dòng được căn giữa đẹp mắt
                customerTypeDisplay.className = "bg-emerald-600/20 text-emerald-400 px-2 py-1 rounded text-xs ml-1 border border-emerald-600/30 inline-block text-center leading-tight align-middle mt-1";
            } else {
                customerTypeDisplay.className = "bg-blue-600/20 text-blue-400 px-2 py-1 rounded text-xs ml-1 border border-blue-600/30 inline-block align-middle";
            }
        }

        // ==================== XE QUẸT VÀO ====================
        if (data.action === "IN") {
            if(mainCamIn) mainCamIn.src = data.img_in;       
            if(imgCamIn) imgCamIn.src = data.img_crop_in;         
            plateInInput.value = data.plate_in;
            
            // Xóa rỗng khu vực Lối Ra để chờ xe tiếp theo
            if(liveCamOut) liveCamOut.src = "https://placehold.co/600x300/0f172a/475569?text=Camera+LPR"; 
            if(imgCamOut) imgCamOut.src = "https://placehold.co/200x80/1a1a1a/475569?text=Waiting...";
            plateOutInput.value = "CHỜ XE RA...";
            
            if(timeInDisplay) timeInDisplay.innerText = data.time_in;
            if(timeOutDisplay) timeOutDisplay.innerText = "--:--:--";
            if(priceDisplay) priceDisplay.innerText = "0 VND";
            if(durationDisplay) durationDisplay.innerText = "Đang gửi...";
            
            plateOutInput.classList.remove('border-red-500', 'text-red-400', 'bg-red-900/20', 'focus:border-red-400');
            plateOutInput.classList.add('border-slate-600', 'focus:border-blue-500', 'bg-slate-800');

            // Xử lý trạng thái thông báo (Badge)
            if (data.warning) {
                statusBadge.innerHTML = '⚠️ ' + data.warning;
                statusBadge.className = 'bg-red-600/20 border border-red-500 text-red-400 font-bold text-center py-3 rounded-lg text-xl mb-6 shadow-[0_0_15px_rgba(239,68,68,0.1)] transition-colors duration-300';
            } else {
                statusBadge.innerHTML = '🕒 ĐANG TRONG BÃI';
                statusBadge.className = 'bg-blue-600/20 border border-blue-500 text-blue-400 font-bold text-center py-3 rounded-lg text-xl mb-6 shadow-[0_0_15px_rgba(59,130,246,0.1)] transition-colors duration-300';
            }

        // ==================== XE QUẸT RA ====================
        } else if (data.action === "OUT") {
            if(liveCamOut) liveCamOut.src = data.img_out;     
            if(imgCamOut) imgCamOut.src = data.img_crop_out;

            // Load lại ảnh lúc vào để bảo vệ đối chiếu
            if(mainCamIn) mainCamIn.src = data.img_in;        
            if(imgCamIn) imgCamIn.src = data.img_crop_in;     

            plateInInput.value = data.plate_in;
            plateOutInput.value = data.plate_out;

            // Thumbnail Footer
            if(recentExitImg) recentExitImg.src = data.img_crop_out;

            if(timeInDisplay) timeInDisplay.innerText = data.time_in;
            if(timeOutDisplay) timeOutDisplay.innerText = data.time_out;
            if(durationDisplay) durationDisplay.innerText = data.duration; 
            
            // Xử lý giá vé 5K
            if (data.customer_type.includes("Khách Đăng Ký")) {
                if(priceDisplay) priceDisplay.innerText = "0 VND (Vé Tháng)";
            } else {
                if(priceDisplay) priceDisplay.innerText = "5,000 VND"; 
            }

            // Xử lý Cảnh báo / Khớp biển
            if (data.warning) {
                // Ưu tiên cảnh báo sai biển số đăng ký Database trước
                statusBadge.innerHTML = '⚠️ ' + data.warning;
                statusBadge.className = 'bg-red-600/20 border border-red-500 text-red-400 font-bold text-center py-3 rounded-lg text-xl mb-6 shadow-[0_0_15px_rgba(239,68,68,0.1)] transition-colors duration-300';
            } else {
                // Nếu DB không báo lỗi, thì check AI cắt biển vào/ra có giống nhau không
                comparePlates();
            }
        }
    };
});