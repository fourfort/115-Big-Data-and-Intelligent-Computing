# find.py（修正版：Line Today 抓 css-xcrf5c + PTT 抓 span[class=""] + 無重複過濾 + 語法修正）
import requests
import time
import random
import hashlib
import os
import pandas as pd
import re
from bs4 import BeautifulSoup


class MovieCommentCrawler:
    def __init__(self, movie_name="我們意外的勇氣"):
        self.movie_name = movie_name
        self.session = requests.Session()
        self.seen_hashes = set()  # 跨來源去重（可選）
        self.data = []
        os.makedirs("data", exist_ok=True)

        self.header_pool = [
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://movie.douban.com/',
                'Accept-Language': 'zh-CN,zh;q=0.9',
                'Connection': 'keep-alive'
            },
            {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Referer': 'https://www.douban.com/',
                'Accept-Language': 'zh-TW,zh;q=0.9',
            },
        ]

    def get_hash(self, text):
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    #切換擷取資料系統之標頭，得以避免被封鎖
    def _get_soup(self, url, params=None, retries=5, delay_base=8): 
        for i in range(retries):
            try:
                headers = random.choice(self.header_pool)
                r = self.session.get(url, headers=headers, params=params, timeout=20)
                if r.status_code == 403:
                    print(f"[!] 403 封鎖，切換 Header + 延遲 {delay_base*(i+1)} 秒...")
                    time.sleep(delay_base * (i + 1))
                    continue
                r.raise_for_status()
                return BeautifulSoup(r.text, 'html.parser')
            except Exception as e:
                print(f"[!] 請求失敗 {url} (第 {i+1} 次): {e}")
                time.sleep(delay_base * (i + 1))
        print(f"[x] 放棄請求: {url}")
        return None

    # === 1. 豆瓣（保持不變）===
    def crawl_douban(self, movie_id="36953756"):
        print("[i] 開始抓取豆瓣短評（status=P），start 0~80")
        total_added = 0
        max_start = 80

        for start in range(0, max_start + 1, 20):
            url = f"https://movie.douban.com/subject/{movie_id}/comments"
            params = {
                'start': start,
                'limit': 20,
                'status': 'P',
                'sort': 'new_score'
            }

            page = (start // 20) + 1
            print(f"[→] 正在抓取第 {page} 頁: start={start}")
            soup = self._get_soup(url, params=params, retries=6, delay_base=15)
            if not soup:
                print(f"[!] start={start} 頁請求失敗，繼續下一頁")
                continue

            items = soup.select('div.comment')
            if not items:
                print(f"[!] start={start} 頁無資料，繼續下一頁")
                continue

            added = 0
            for item in items:
                content_elem = item.select_one('span.short') or item.select_one('span.all')
                if not content_elem:
                    continue
                content = content_elem.get_text(strip=True)
                if len(content) < 3:
                    continue

                # 評分
                rating_elem = item.select_one('span[class*="allstar"]')
                rating = 0
                if rating_elem:
                    cls = rating_elem['class'][0]
                    if cls.startswith('allstar'):
                        rating = int(cls.replace('allstar', '')) // 10

                date_elem = item.select_one('span.comment-time')
                date = date_elem.get('title', '').strip() if date_elem else ''

                # 去重：內容+日期
                h = self.get_hash(content + date)
                if h in self.seen_hashes:
                    continue
                self.seen_hashes.add(h)

                self.data.append({
                    'source': 'douban',
                    'content': content,
                    'rating': rating,
                    'date': date,
                    'url': f"{url}?start={start}"
                })
                added += 1
                total_added += 1

            print(f"[+] 豆瓣正向 start={start}: +{added} 筆（共 {total_added}）")
            time.sleep(random.uniform(18, 28))

        print(f"[*] 豆瓣短評結束，共抓取 {total_added} 筆")

    # === 2. PTT：修正推文重複抓取問題 ===
    def crawl_ptt(self):
        ptt_urls = ["https://www.pttweb.cc/bbs/movie/M.1761506965.A.45C","https://www.pttweb.cc/bbs/Gossiping/M.1760022805.A.AAA"]
        print("[i] 開始抓取 PTT 特定 URL（內文 + 推文，防重複）...")
        total_added = 0
        seen_push_hashes = set()  # 專門用於 PTT 推文去重
        for idx, url in enumerate(ptt_urls, 1):
            print(f"[→] 抓取 PTT URL {idx}: {url}")
            soup = self._get_soup(url, retries=5, delay_base=8)
            if not soup:
                print(f"[!] PTT URL {idx} 請求失敗，跳過")
                continue
            added_this_page = 0
            # === 主內文：抓 <span class=""> ===
            main_span = soup.select_one('span[class=""]')
            if main_span:
                text = main_span.get_text(separator='\n')
                text = re.sub(r'--.*', '', text, flags=re.DOTALL).strip()
                text = re.sub(r'〓[^〓]*?不對〓', '', text)
                text = re.sub(r'[^\u4e00-\u9fff\w\s.,!?]', ' ', text).strip()
                if len(text) > 20:
                    h = self.get_hash(f"main_{text}_{url}")
                    if h not in self.seen_hashes:
                        self.seen_hashes.add(h)
                        self.data.append({
                            'source': 'ptt_main',
                            'content': text,
                            'rating': None,
                            'date': '',
                            'url': url
                        })
                        total_added += 1
                        added_this_page += 1
                        print(f"[+] PTT 主內文: +1 筆（{len(text)} 字）")
            # === 推文：改用「最深層」的 div + 內容去重 ===
            # 策略：只抓「沒有子 div[data-v-9fbc241a]」的 div → 最內層
            push_divs = soup.find_all('div', {'data-v-9fbc241a': True})
            leaf_divs = [div for div in push_divs if not div.find('div', {'data-v-9fbc241a': True})]
            push_count = 0
            for div in leaf_divs:
                push_span = div.select_one('span[class=""]')
                if not push_span:
                    continue
                raw = push_span.get_text(strip=True)
                # 過濾垃圾
                if '〓' in raw and '不對' in raw:
                    continue
                content = re.sub(r'〓[^〓]*?不對〓', '', raw)
                content = re.sub(r'[^\u4e00-\u9fff\w\s.,!?]', ' ', content).strip()
                if len(content) < 2:
                    continue
                # 關鍵：用內容 + URL 做去重
                push_hash = self.get_hash(content + url)
                if push_hash in seen_push_hashes:
                    continue
                seen_push_hashes.add(push_hash)
                self.data.append({
                    'source': 'ptt_push',
                    'content': content,
                    'rating': None,
                    'date': '',
                    'url': url
                })
                total_added += 1
                added_this_page += 1
                push_count += 1
                print(f"[+] PTT 推文: +1 筆 → {content[:40]}...")
            print(f"[*] PTT URL {idx} 結束：主文 + 推文 共 +{added_this_page} 筆（累計 {total_added}）")
        print(f"[*] PTT 特定URL結束，共抓取 {total_added} 筆（已去重）")

    # === 3. Line Today ===
    def crawl_line_today(self, txt_path="data/line_today_comments.txt"):
        print("[i] 開始讀取 Line Today 本地評論檔案...")
        total_added = 0

        if not os.path.exists(txt_path):
            print(f"[!] 檔案不存在: {txt_path}，Line Today 跳過")
            return

        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]

            for text in lines:
                if len(text) < 3:
                    continue

                # 去重：內容 + 來源
                h = self.get_hash(text + "line_today_local")
                if h in self.seen_hashes:
                    continue
                self.seen_hashes.add(h)

                self.data.append({
                    'source': 'line_today',
                    'content': text,
                    'rating': None,
                    'date': '',
                    'url': 'local_file'
                })
                total_added += 1
                print(f"[+] Line Today 本地評論: +1 筆 → {text[:40]}...")

            print(f"[*] Line Today 本地檔案讀取完成，共 {total_added} 筆")

        except Exception as e:
            print(f"[!] 讀取 Line Today TXT 失敗: {e}")

    # === 4. 儲存 ===
    def save(self, filepath="data/raw_comments.csv"):
        df = pd.DataFrame(self.data)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n=== 爬蟲完成 ===")
        print(f"總筆數: {len(df)}")
        print(f"來源分布:\n{df['source'].value_counts()}")
        return df

