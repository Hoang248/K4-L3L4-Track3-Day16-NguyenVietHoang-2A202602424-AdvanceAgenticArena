"""LỚP `retry` — bài giảng Day 16, §7 (Failure Handling & Retries).

NHIỆM VỤ: tầng công cụ hỏng có chủ ý (~15% lượt gọi), và mô hình xử lý sai
theo hai nửa — nửa sau mới là nửa đắt:

  (a) Với NOISE — kiểu hỏng ồn ào nhất — mô hình gọi lại y hệt lượt cũ tối
      đa hai lần, mỗi lần tốn trọn một vòng gọi model, rồi bỏ cuộc mà
      KHÔNG có nội dung.
  (b) Với mọi kiểu hỏng còn lại — bị cắt, timeout, không tìm thấy tài
      liệu, biểu thức sai — mô hình KHÔNG NHẬN RA GÌ CẢ. Nó đi tiếp và
      lặng lẽ trả lời bằng một tài liệu nó chưa từng đọc.

Thử lại ở BÊN DƯỚI mô hình, trong `wrap_tool_call`, sửa cả hai: nửa (a)
không còn tốn vòng gọi model nào, nửa (b) biến mất.

TÍN HIỆU — dùng `arena.model.is_degraded`, tức là TOÀN BỘ tập
`DEGRADED_MARKERS`, chứ không phải mỗi cái marker mà bản thân mô hình phản
ứng. Đúng chỗ khác nhau đó chính là giá trị của lớp này:

    (not result.ok) or is_degraded(result.content)

`ok=True` KHÔNG có nghĩa là ổn: bản bị cắt và bản nhiễu đều về với
`ok=True`. Đó là cái bẫy.

Thử lại có tác dụng vì tầng công cụ khoá xác suất hỏng theo
`(seed, số thứ tự lượt gọi)`, nên lượt gọi lại rơi vào một chỉ số MỚI và
được tung lại độc lập.

ĐỌC KỸ — VÌ SAO LỚP NÀY TRÔNG NHƯ KHÔNG CHẠY:

**Cắm riêng nó lên baseline, `retry` đo được -0.35 (5 seed gốc; +0.19 ở
20 seed) và chỉ thắng baseline ở 20/120 lượt chạy.** Đó không phải lỗi
cài đặt của bạn. Không có `citation_checker` thì bằng chứng mà `retry`
cứu về vẫn bị lỗi trích dẫn sai của mô hình vứt đi, nên nó chẳng mua được
gì mà vẫn tốn một lượt công cụ. Tiêu chí nghiệm thu vì thế là
LEAVE-ONE-OUT: rút `retry` ra khỏi full stack thì điểm TỤT XUỐNG.

**Sản phẩm thật của lớp này là PHƯƠNG SAI, không phải trung bình.** Trên
30 lượt chạy (6 brief x 5 seed gốc), nó kéo độ lệch chuẩn của tổng điểm
từ 24.21 xuống 11.43, và số quan sát hỏng lọt tới mô hình từ 30 xuống 2.
Trong một cuộc thi chấm trên vài brief, giảm một nửa độ dao động đáng giá
hơn một điểm trung bình: đó là khác biệt giữa một bài chắc chắn và một
bài may mắn.

ĐỪNG THỬ LẠI VÔ HẠN, VÀ ĐỪNG THỬ LẠI BẰNG LƯỢT DÀNH CHO `submit`: mỗi lần
gọi lại tốn một lượt trong ngân sách công cụ. `budget_policy` KHÔNG cứu
được bạn ở đây — hook `wrap_tool_call` của nó nằm NGOÀI vòng lặp thử lại
của bạn, nên nó chỉ thấy lượt gọi đầu tiên. Một lớp `retry` không tự kiểm
tra ngân sách làm cả stack tiêu lố: đo được 34/120 lượt chạy kết thúc ở 9+
lượt gọi trong khi brief cho 8, và efficiency tụt từ 14.24 xuống 12.06.

CÔNG CỤ CÓ SẴN:
    from arena.model import is_degraded
    ctx.state           -> dict tuỳ bạn dùng để đếm số lần thử lại
    ctx.tools.calls     -> số lượt gọi công cụ đã dùng (kể cả submit)
    ctx.max_tool_calls  -> ngân sách của brief, hoặc None

Cài đặt:  ReActAgent(..., middleware=[..., Retry()])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from arena.model import is_degraded

from harness.middleware import Middleware

#: Tổng số lần thử, tính cả lần đầu.
DEFAULT_MAX_ATTEMPTS = 3

#: Số lượt để dành cho `submit` mà agent vẫn còn phải gọi.
DEFAULT_RESERVE = 1


class Retry(Middleware):
    """Gọi lại một lượt công cụ trả về kết quả hỏng hoặc suy giảm."""

    name = "retry"

    def __init__(
        self,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        reserve: int = DEFAULT_RESERVE,
    ) -> None:
        self.max_attempts = max(1, int(max_attempts))
        self.reserve = max(0, int(reserve))

    def wrap_tool_call(self, ctx, call, name, args):
        result = call(name, args)
        attempts = 1
        while attempts < self.max_attempts and (not result.ok or is_degraded(result.content)):
            limit = ctx.max_tool_calls
            if limit is not None and ctx.tools.calls >= limit - self.reserve:
                break
            result = call(name, args)
            attempts += 1
        ctx.state["retry_attempts"] = ctx.state.get("retry_attempts", 0) + attempts - 1
        return result
