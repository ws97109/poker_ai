import pytesseract
from PIL import ImageGrab, Image
import cv2
import numpy as np
import pyautogui
from keras.models import load_model

# 設置 Tesseract 的路徑，這個路徑需要指向你安裝的 Tesseract 執行檔
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 定義撲克牌的數值映射，將撲克牌的字母數值（例如A, K, Q, J）轉換為數字
card_mapping = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
    '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 11, 'Q': 12, 'K': 13, 'A': 14
}

# 定義花色映射，將撲克牌的花色轉換為數值
suits_mapping = {
    '♠': 1, '♥': 2, '♦': 3, '♣': 4
}

def card_to_numeric(card):
    """
    將單張牌轉換為數值，例如 'A♠' 轉換為 (14, 1)
    :param card: 例如 'A♠', '10♥'
    :return: 牌的數值與花色數值
    """
    value, suit = card[:-1], card[-1]  # 分離牌面和花色
    return card_mapping[value], suits_mapping[suit]  # 轉換為對應數字

def capture_screen(bbox=None):
    """
    使用 Pillow 截取屏幕，bbox 指定要截取的螢幕區域
    :param bbox: (x1, y1, x2, y2)，指定螢幕區域
    :return: 截取到的螢幕圖片
    """
    screenshot = ImageGrab.grab(bbox=bbox)  # 截取指定範圍的螢幕
    return screenshot

def process_image_for_ocr(image):
    """
    對截取的圖片進行預處理，以提高 OCR 的準確度
    :param image: 從螢幕截取到的圖片
    :return: 經過處理後的灰度圖與二值化圖片
    """
    # 將圖片轉換為灰度圖，這樣有助於提取文字
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    
    # 二值化處理，將灰度圖片轉換為黑白圖片
    processed_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)[1]
    
    return processed_image  # 返回處理後的圖片

def extract_text_from_image(image):
    """
    使用 Tesseract OCR 來提取圖像中的文本
    :param image: 經過處理的圖片
    :return: 從圖片中提取的文字
    """
    processed_image = process_image_for_ocr(image)  # 首先對圖像進行處理
    
    # 使用 Tesseract 進行文字提取
    text = pytesseract.image_to_string(processed_image)
    
    return text  # 返回提取的文字

def preprocess_game_state(hand_cards, community_cards, current_bet, total_chips):
    """
    將所有遊戲狀態數據轉換為神經網絡的輸入向量
    :param hand_cards: 玩家手牌的數值形式
    :param community_cards: 社區牌的數值形式
    :param current_bet: 當前賭注金額
    :param total_chips: 玩家剩餘籌碼
    :return: 轉換好的輸入向量
    """
    # 將手牌和社區牌轉換為數值形式
    numeric_hand = [card_to_numeric(card) for card in hand_cards]
    numeric_community = [card_to_numeric(card) for card in community_cards]

    # 展平成一個向量，包含手牌與社區牌的數值
    game_state_vector = []
    for card in numeric_hand + numeric_community:
        game_state_vector.extend(card)  # 把牌值和花色都加入向量
    
    # 加入賭注和籌碼數據
    game_state_vector.append(current_bet)
    game_state_vector.append(total_chips)
    
    return np.array(game_state_vector)  # 返回轉換好的輸入數據

def perform_action(action):
    """
    根據 AI 的預測行動來模擬點擊撲克軟體的按鈕
    :param action: AI 的決策，例如 'raise', 'fold', 'call'
    """
    if action == 'raise':
        # 模擬點擊加注按鈕，假設按鈕位於座標 (1000, 500)
        pyautogui.moveTo(1000, 500)
        pyautogui.click()
    elif action == 'fold':
        # 模擬點擊棄牌按鈕，假設按鈕位於座標 (800, 500)
        pyautogui.moveTo(800, 500)
        pyautogui.click()
    elif action == 'call':
        # 模擬點擊跟注按鈕，假設按鈕位於座標 (900, 500)
        pyautogui.moveTo(900, 500)
        pyautogui.click()

if __name__ == '__main__':
    # 載入已經訓練好的 AI 模型
    model = load_model('path_to_your_model.h5')

    # 假設要截取撲克桌面的某個區域
    bbox = (100, 100, 500, 400)  # 這些座標根據撲克桌面調整
    screenshot = capture_screen(bbox=bbox)  # 截圖撲克桌面

    # 使用 OCR 從截圖中提取牌面、賭注等信息
    extracted_text = extract_text_from_image(screenshot)
    print("提取的文字:", extracted_text)

    # 模擬一個手牌和社區牌的例子，這些數據應來自 OCR 的結果
    hand_cards = ["A♠", "K♠"]
    community_cards = ["10♣", "J♥", "Q♠"]
    current_bet = 500  # 假設目前的賭注
    total_chips = 1000  # 假設玩家的總籌碼數

    # 將遊戲狀態預處理為神經網絡的輸入向量
    input_vector = preprocess_game_state(hand_cards, community_cards, current_bet, total_chips)

    # 將輸入向量 reshape 為 (1, n) 的形式餵入神經網絡
    input_vector = input_vector.reshape(1, -1)
    predicted_action = model.predict(input_vector)  # 進行 AI 預測

    # 將模型的輸出結果轉化為具體行動
    action = np.argmax(predicted_action)  # 假設模型輸出加注、跟注、棄牌三種行動
    
    # 模擬執行 AI 的行動，根據 AI 的預測結果來點擊按鈕
    if action == 0:
        perform_action('raise')
    elif action == 1:
        perform_action('call')
    elif action == 2:
        perform_action('fold')

    # 打印 AI 的決策結果
    print(f"AI 決定進行: {['raise', 'call', 'fold'][action]}")
