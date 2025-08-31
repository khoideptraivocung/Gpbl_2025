import google.generativeai as genai


GEMINI_API_KEY = "AIzaSyBG7BjXOCV8zuAvxRjuh2wA8NyF3N1Lg-8"
genai.configure(api_key=GEMINI_API_KEY)

# --- Khởi tạo model ---
model = genai.GenerativeModel("gemini-1.5-flash")

def ask_gemini(prompt: str) -> str:
    """Hàm gửi prompt tới Gemini và nhận kết quả"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Lỗi khi gọi Gemini: {e}"

def main():
    print("Chat với Gemini (gõ 'exit' để thoát)")
    while True:
        user_input = input("Bạn> ")
        if user_input.lower() in ["exit", "quit"]:
            break
        answer = ask_gemini(user_input)
        print("Gemini>", answer)

if __name__ == "__main__":
    main()
