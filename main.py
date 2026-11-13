import os
from fastapi import FastAPI, Request, Response, BackgroundTasks
import requests
from dotenv import load_dotenv
from google import genai
from PIL import Image
import io
import json
import cv2
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv()

app = FastAPI()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

ADMIN_CHAT_ID_1 = os.getenv("TELEGRAM_ADMIN_CHAT_ID1")
ADMIN_CHAT_ID_2 = os.getenv("TELEGRAM_ADMIN_CHAT_ID2")
ADMIN_CHAT_IDS = [chat_id for chat_id in [ADMIN_CHAT_ID_1, ADMIN_CHAT_ID_2] if chat_id]

client = genai.Client(api_key=GEMINI_API_KEY)

table_rows = []
active_admin_message_ids = {}

def reset_daily_table():
    global table_rows, active_admin_message_ids
    table_rows.clear()
    active_admin_message_ids.clear()
    print("Daily payment table has been automatically reset at midnight.")

scheduler = BackgroundScheduler()
scheduler.add_job(reset_daily_table, 'cron', hour=0, minute=0)
scheduler.start()


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    message = data.get("message") or data.get("edited_message")
    
    if message:
        chat_id = message["chat"]["id"]
        text = message.get("text", "")

        if text.startswith("/export") and str(chat_id) in ADMIN_CHAT_IDS:
            background_tasks.add_task(send_excel_export, chat_id)
            return Response(status_code=200)

        if "photo" in message:
            sender = message.get("from", {})
            sender_name = sender.get("first_name", "Unknown User")
            sender_username = f"@{sender.get('username')}" if sender.get("username") else "None"

            photos = message["photo"]
            best_photo = photos[-1]
            file_id = best_photo["file_id"]

            background_tasks.add_task(
                process_payment_with_table_append, 
                file_id, 
                sender_name, 
                sender_username
            )

    return Response(status_code=200)


def blur_sensitive_areas(image_bytes):
    """OpenCV privacy mask for sensitive sections"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is not None:
        h, w, _ = img.shape
        img[0:int(h * 0.20), 0:w] = cv2.GaussianBlur(img[0:int(h * 0.20), 0:w], (51, 51), 0)
        img[int(h * 0.80):h, 0:w] = cv2.GaussianBlur(img[int(h * 0.80):h, 0:w], (51, 51), 0)

    _, encoded_img = cv2.imencode('.jpg', img)
    return encoded_img.tobytes()


def process_payment_with_table_append(file_id, sender_name, sender_username):
    global table_rows, active_admin_message_ids
    
    try:
        file_info_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}"
        file_response = requests.get(file_info_url).json()

        if not file_response.get("ok"):
            return

        file_path = file_response["result"]["file_path"]
        image_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"

        img_response = requests.get(image_url)
        raw_image_bytes = img_response.content

        sanitized_bytes = blur_sensitive_areas(raw_image_bytes)
        img = Image.open(io.BytesIO(sanitized_bytes))

        current_date = datetime.now().strftime("%Y-%m-%d")
    
        prompt = (
            f"Analyze this payment receipt image and extract only the following 5 details in strict JSON format. "
            f"Do not include personal identifiable info. If date is missing on receipt, use '{current_date}'. "
            f"If note or remark or message is missing on receipt, output '-' for the note field. "
            "JSON keys required: "
            "1. amount (numerical value only, integer format without any decimals or trailing zeros like .00, and without any minus signs) "
            "2. currency (always output currency strictly as 'KS' regardless of how it appears on the receipt) "
            "3. transaction_id (Ref No) "
            "4. date (Transaction date in YYYY-MM-DD format) "
            "5. note (Any remark, note, description, or message written on the receipt, or '-' if none)"
        )

        models_to_try = [
            'models/gemini-3.6-flash',
            'models/gemini-3.5-flash-lite',
            'models/gemini-3.1-flash-lite',
            'models/gemma-4-26b-a4b-it',
            'models/gemini-3.8-flash'
        ]
        
        response = None
        success = False

        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name, 
                    contents=[prompt, img]
                )
                success = True
                break
            except Exception:
                continue

        if not success:
            raise Exception("All Gemini models are currently experiencing high demand (503/429 error).")

        text_response = response.text.strip()
        text_response = text_response.replace("```json", "").replace("```", "").strip()
        
        extracted_data = json.loads(text_response)
        amount = extracted_data.get("amount", "0")
        currency = extracted_data.get("currency", "KS")
        transaction_id = extracted_data.get("transaction_id", "N/A")
        tx_date = extracted_data.get("date", current_date)
        note = extracted_data.get("note", "-")

        new_row = f"| {sender_name} | {amount} {currency} | `{transaction_id}` | {tx_date} | {note} |"
        table_rows.append(new_row)

        total_sum = 0
        for r in table_rows:
            parts = [p.strip() for p in r.split('|') if p.strip() != '']
            if len(parts) >= 2:
                amt_clean = parts[1].replace("KS", "").replace(",", "").strip()
                try:
                    total_sum += float(amt_clean)
                except ValueError:
                    pass

        table_header = (
            "**Live Payment Table**\n\n"
            "| Sender | Amount | Transaction ID | Date | Note |\n"
        )
        table_body = "\n".join(table_rows)
        table_footer = f"\n| **Total** | **{int(total_sum):,} KS** | | | |"
        
        full_table_text = table_header + table_body + table_footer

        for admin_chat_id in ADMIN_CHAT_IDS:
            current_msg_id = active_admin_message_ids.get(admin_chat_id)
            updated_successfully = False
            
            if current_msg_id is not None:
                edit_res = requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageText",
                    json={
                        "chat_id": admin_chat_id,
                        "message_id": current_msg_id,
                        "text": full_table_text,
                        "parse_mode": "Markdown"
                    }
                ).json()
                
                if edit_res.get("ok"):
                    updated_successfully = True

            if not updated_successfully:
                res = requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                    json={
                        "chat_id": admin_chat_id,
                        "text": full_table_text,
                        "parse_mode": "Markdown",
                        "disable_notification": True
                    }
                ).json()
                
                if res.get("ok"):
                    active_admin_message_ids[admin_chat_id] = res["result"]["message_id"]

    except Exception as e:
        print(f"Background Processing Error: {e}")


def send_excel_export(chat_id):
    """Export table rows to Excel including Note column"""
    global table_rows 
    
    if not table_rows:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": "No transaction records available to export today."}
        )
        return

    data_list = []
    total_amount = 0.0  
    
    for row in table_rows:
        parts = [p.strip() for p in row.split('|') if p.strip() != '']
        if len(parts) >= 5:
            amount_str = parts[1].replace("KS", "").replace(",", "").strip()
            try:
                numeric_amount = float(amount_str)
            except ValueError:
                numeric_amount = 0.0

            total_amount += numeric_amount  

            data_list.append({
                "Sender": parts[0],
                "Amount": numeric_amount,
                "Transaction ID": parts[2].replace("`", ""),
                "Date": parts[3],
                "Note": parts[4]
            })

    df = pd.DataFrame(data_list)
    
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    file_path = f"transactions_report_{current_date_str}.xlsx"
    
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Transactions')

    wb = openpyxl.load_workbook(file_path)
    ws = wb['Transactions']

    header_font = Font(name='Arial', size=11, bold=True, color='000000')
    header_alignment = Alignment(horizontal='center', vertical='center')

    data_font = Font(name='Arial', size=10)
    bold_font = Font(name='Arial', size=10, bold=True)
    thin_border = Border(
        left=Side(style='thin', color='D9D9D9'),
        right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'),
        bottom=Side(style='thin', color='D9D9D9')
    )

    ws.row_dimensions[1].height = 25
    for col_num in range(1, len(df.columns) + 1):
        cell = ws.cell(row=1, column=col_num)
        cell.font = header_font
        cell.alignment = header_alignment
        cell.border = thin_border

    for row_num in range(2, len(df) + 2):
        ws.row_dimensions[row_num].height = 20
        for col_num in range(1, len(df.columns) + 1):
            cell = ws.cell(row=row_num, column=col_num)
            cell.font = data_font
            cell.border = thin_border
            
            if col_num == 1:
                cell.alignment = Alignment(horizontal='left', vertical='center')
            elif col_num == 2:
                cell.alignment = Alignment(horizontal='right', vertical='center')
                cell.number_format = '#,##0 "KS"'
            else:
                cell.alignment = Alignment(horizontal='center', vertical='center')

    last_row = len(df) + 2
    ws.row_dimensions[last_row].height = 22

    label_cell = ws.cell(row=last_row, column=1, value="Total Sum")
    label_cell.font = bold_font
    label_cell.alignment = Alignment(horizontal='left', vertical='center')
    label_cell.border = thin_border

    sum_cell = ws.cell(row=last_row, column=2)
    sum_cell.value = total_amount  
    sum_cell.font = bold_font
    sum_cell.alignment = Alignment(horizontal='right', vertical='center')
    sum_cell.number_format = '#,##0 "KS"'
    sum_cell.border = thin_border

    for col_num in range(3, len(df.columns) + 1):
        empty_cell = ws.cell(row=last_row, column=col_num, value="")
        empty_cell.border = thin_border

    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            if cell.value:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = max(max_len + 6, 18)

    wb.save(file_path)

    with open(file_path, "rb") as f:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument",
            data={"chat_id": chat_id, "caption": f"Daily Transaction Report ({current_date_str})"},
            files={"document": f}
        )