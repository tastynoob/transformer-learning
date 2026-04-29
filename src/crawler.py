# 帮我生成一批AI训练用的文本数据，用json文件格式保存

import requests
import json
import time
import random
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
from datetime import datetime

class TextDataCrawler:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.data = []
        
    def crawl_wikipedia_articles(self, topics, max_articles=10):
        """爬取维基百科文章"""
        base_url = "https://zh.wikipedia.org/wiki/"
        
        for topic in topics:
            try:
                url = base_url + topic
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # 提取标题
                    title = soup.find('h1', class_='firstHeading')
                    title_text = title.text if title else topic
                    
                    # 提取正文段落
                    content_div = soup.find('div', class_='mw-parser-output')
                    paragraphs = content_div.find_all('p') if content_div else []
                    
                    content = ""
                    for p in paragraphs[:5]:  # 只取前5段
                        text = p.get_text().strip()
                        if len(text) > 50:  # 过滤太短的段落
                            content += text + "\n\n"
                    
                    if content:
                        self.data.append({
                            "id": len(self.data) + 1,
                            "source": "wikipedia",
                            "title": title_text,
                            "content": content.strip(),
                            "category": "encyclopedia",
                            "timestamp": datetime.now().isoformat(),
                            "url": url
                        })
                        
                    print(f"已爬取: {title_text}")
                    time.sleep(random.uniform(1, 3))  # 随机延迟
                    
            except Exception as e:
                print(f"爬取 {topic} 失败: {str(e)}")
                continue
                
        return len([item for item in self.data if item["source"] == "wikipedia"])
    
    def crawl_news_sites(self, urls, max_articles=20):
        """爬取新闻网站"""
        for url in urls:
            try:
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # 通用的新闻文章选择器
                    article_selectors = [
                        'article',
                        '.article-content',
                        '.news-content',
                        '.content',
                        '[class*="article"]',
                        '[class*="news"]'
                    ]
                    
                    title = soup.find('h1') or soup.find('title')
                    title_text = title.get_text().strip() if title else "无标题"
                    
                    content = ""
                    for selector in article_selectors:
                        elements = soup.select(selector)
                        for element in elements:
                            paragraphs = element.find_all(['p', 'div'])
                            for p in paragraphs:
                                text = p.get_text().strip()
                                if len(text) > 30:
                                    content += text + "\n\n"
                            if content:
                                break
                        if content:
                            break
                    
                    if content and len(content) > 100:
                        self.data.append({
                            "id": len(self.data) + 1,
                            "source": "news",
                            "title": title_text,
                            "content": content.strip(),
                            "category": "news",
                            "timestamp": datetime.now().isoformat(),
                            "url": url
                        })
                        
                    print(f"已爬取新闻: {title_text[:50]}...")
                    time.sleep(random.uniform(2, 4))
                    
            except Exception as e:
                print(f"爬取 {url} 失败: {str(e)}")
                continue
    
    def generate_qa_pairs(self):
        """生成问答对数据"""
        qa_data = [
            {
                "id": len(self.data) + 1,
                "source": "generated",
                "question": "什么是人工智能？",
                "answer": "人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
                "category": "qa",
                "timestamp": datetime.now().isoformat()
            },
            {
                "id": len(self.data) + 2,
                "source": "generated", 
                "question": "机器学习的主要类型有哪些？",
                "answer": "机器学习主要分为三类：监督学习（使用标记数据）、无监督学习（发现数据中的模式）和强化学习（通过试错学习）。",
                "category": "qa",
                "timestamp": datetime.now().isoformat()
            },
            {
                "id": len(self.data) + 3,
                "source": "generated",
                "question": "深度学习与传统机器学习的区别是什么？",
                "answer": "深度学习使用多层神经网络自动学习特征表示，而传统机器学习通常需要手工设计特征。深度学习在处理大量数据时表现更好。",
                "category": "qa", 
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        self.data.extend(qa_data)
        return len(qa_data)
    
    def add_synthetic_text(self):
        """添加合成文本数据"""
        synthetic_texts = [
            {
                "id": len(self.data) + 1,
                "source": "synthetic",
                "title": "科技发展趋势",
                "content": "随着科技的快速发展，人工智能、物联网、区块链等新兴技术正在改变我们的生活方式。这些技术的融合将为未来社会带来更多可能性。企业需要适应这种变化，投资于数字化转型以保持竞争力。",
                "category": "technology",
                "timestamp": datetime.now().isoformat()
            },
            {
                "id": len(self.data) + 2,
                "source": "synthetic",
                "title": "环境保护的重要性",
                "content": "环境保护是当今世界面临的重要挑战之一。气候变化、污染和生物多样性丧失等问题需要全球合作来解决。每个人都应该采取行动，减少碳足迹，保护我们共同的地球家园。",
                "category": "environment",
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        self.data.extend(synthetic_texts)
        return len(synthetic_texts)
    
    def save_to_json(self, filename="training_data.json"):
        """保存数据到JSON文件"""
        output_path = os.path.join(os.getcwd(), filename)
        
        # 添加元数据
        output_data = {
            "metadata": {
                "total_records": len(self.data),
                "created_at": datetime.now().isoformat(),
                "categories": list(set(item.get("category", "unknown") for item in self.data)),
                "sources": list(set(item.get("source", "unknown") for item in self.data))
            },
            "data": self.data
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"数据已保存到: {output_path}")
            print(f"总共生成 {len(self.data)} 条训练数据")
            return True
        except Exception as e:
            print(f"保存文件失败: {str(e)}")
            return False
    
    def run_crawler(self):
        """运行爬虫主程序"""
        print("开始生成AI训练文本数据...")
        
        # 1. 爬取维基百科文章
        wiki_topics = [
            "人工智能", "机器学习", "深度学习", "神经网络", "自然语言处理",
            "计算机视觉", "数据科学", "算法", "程序设计", "软件工程"
        ]
        wiki_count = self.crawl_wikipedia_articles(wiki_topics, max_articles=10)
        print(f"维基百科文章: {wiki_count} 篇")
        
        # 2. 生成问答对
        qa_count = self.generate_qa_pairs()
        print(f"问答对: {qa_count} 对")
        
        # 3. 添加合成文本
        synthetic_count = self.add_synthetic_text()
        print(f"合成文本: {synthetic_count} 篇")
        
        # 4. 保存数据
        self.save_to_json()
        
        return len(self.data)

def main():
    crawler = TextDataCrawler()
    total_records = crawler.run_crawler()
    print(f"数据生成完成！总共 {total_records} 条记录")

if __name__ == "__main__":
    main()