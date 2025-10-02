"""
Example: Chinese Document Processing in TextReadingRAG

This example demonstrates how to process Traditional Chinese documents
and perform Chinese language queries.
"""

import asyncio
from pathlib import Path

from src.core.config import Settings
from src.rag.language_utils import (
    detect_language,
    is_chinese,
    split_chinese_text,
    tokenize_chinese,
)


def example_language_detection():
    """Demonstrate language detection."""
    print("\n=== Language Detection Example ===\n")

    texts = {
        "English": "This is an English document about artificial intelligence.",
        "Traditional Chinese": "這是一份關於人工智能的繁體中文文檔。",
        "Simplified Chinese": "这是一份关于人工智能的简体中文文档。",
        "Mixed": "This document contains both English and 中文内容.",
    }

    for name, text in texts.items():
        lang = detect_language(text)
        is_cn = is_chinese(text)
        print(f"{name}:")
        print(f"  Text: {text}")
        print(f"  Detected: {lang}")
        print(f"  Is Chinese: {is_cn}")
        print()


def example_chinese_tokenization():
    """Demonstrate Chinese text tokenization."""
    print("\n=== Chinese Tokenization Example ===\n")

    texts = [
        "我喜歡閱讀技術文檔",
        "人工智能技術正在快速發展",
        "機器學習是人工智能的重要分支",
    ]

    for text in texts:
        tokens = tokenize_chinese(text)
        print(f"Original: {text}")
        print(f"Tokens:   {' | '.join(tokens)}")
        print(f"Count:    {len(tokens)} tokens")
        print()


def example_chinese_text_splitting():
    """Demonstrate Chinese text splitting."""
    print("\n=== Chinese Text Splitting Example ===\n")

    # Sample Traditional Chinese text about AI
    text = """
    人工智能是計算機科學的一個分支。它致力於創建能夠執行需要人類智能的任務的系統。
    機器學習是人工智能的核心技術之一。它使計算機能夠從數據中學習並改進性能。
    深度學習是機器學習的一個子領域。它使用多層神經網絡來處理複雜的數據。
    自然語言處理是人工智能的另一個重要分支。它使計算機能夠理解和生成人類語言。
    計算機視覺技術使機器能夠理解和解釋視覺信息。這在自動駕駛和醫療診斷中有重要應用。
    """

    text = text.strip()

    # Split with Chinese-optimized parameters
    chunks = split_chinese_text(
        text,
        chunk_size=100,  # 100 Chinese characters
        chunk_overlap=20   # 20 character overlap
    )

    print(f"Original text length: {len(text)} characters")
    print(f"Number of chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i} ({len(chunk)} chars):")
        print(f"  {chunk}")
        print()


def example_multilingual_comparison():
    """Compare English and Chinese text processing."""
    print("\n=== Multilingual Comparison Example ===\n")

    english_text = """
    Artificial intelligence is a branch of computer science. It aims to create systems
    that can perform tasks requiring human intelligence. Machine learning is one of the
    core technologies of AI. It enables computers to learn from data and improve performance.
    """

    chinese_text = """
    人工智能是計算機科學的一個分支。它致力於創建能夠執行需要人類智能的任務的系統。
    機器學習是人工智能的核心技術之一。它使計算機能夠從數據中學習並改進性能。
    """

    # Process English
    en_lang = detect_language(english_text)
    en_chunks = split_chinese_text(english_text.strip(), chunk_size=150, chunk_overlap=30)

    # Process Chinese
    zh_lang = detect_language(chinese_text)
    zh_chunks = split_chinese_text(chinese_text.strip(), chunk_size=100, chunk_overlap=20)

    print("English Processing:")
    print(f"  Language: {en_lang}")
    print(f"  Text length: {len(english_text.strip())} chars")
    print(f"  Chunks: {len(en_chunks)}")
    print(f"  Avg chunk size: {sum(len(c) for c in en_chunks) / len(en_chunks):.1f} chars")
    print()

    print("Chinese Processing:")
    print(f"  Language: {zh_lang}")
    print(f"  Text length: {len(chinese_text.strip())} chars")
    print(f"  Chunks: {len(zh_chunks)}")
    print(f"  Avg chunk size: {sum(len(c) for c in zh_chunks) / len(zh_chunks):.1f} chars")
    print()

    print("Analysis:")
    print(f"  Chinese has ~{len(chinese_text.strip()) / len(english_text.strip()):.2f}x the information density")
    print(f"  That's why Chinese chunk_size (100) < English chunk_size (150)")


def example_tokenizer_comparison():
    """Compare English and Chinese tokenization."""
    print("\n=== Tokenizer Comparison Example ===\n")

    english = "Machine learning is a subset of artificial intelligence"
    chinese = "機器學習是人工智能的子集"

    # English tokenization (simple split)
    en_tokens = english.lower().split()

    # Chinese tokenization (jieba)
    zh_tokens = tokenize_chinese(chinese)

    print("English:")
    print(f"  Text: {english}")
    print(f"  Tokens: {en_tokens}")
    print(f"  Count: {len(en_tokens)}")
    print()

    print("Chinese:")
    print(f"  Text: {chinese}")
    print(f"  Tokens: {zh_tokens}")
    print(f"  Count: {len(zh_tokens)}")
    print()

    print("Comparison:")
    print(f"  Similar semantic content")
    print(f"  English: {len(english)} chars, {len(en_tokens)} tokens")
    print(f"  Chinese: {len(chinese)} chars, {len(zh_tokens)} tokens")
    print(f"  Chinese is {len(chinese) / len(english):.2f}x more compact")


def example_real_world_scenario():
    """Demonstrate a real-world scenario with mixed documents."""
    print("\n=== Real-World Scenario Example ===\n")

    documents = [
        {
            "title": "Introduction to AI",
            "content": "Artificial intelligence (AI) refers to the simulation of human intelligence in machines."
        },
        {
            "title": "人工智能簡介",
            "content": "人工智能是指機器對人類智能的模擬。它包括學習、推理和自我修正等能力。"
        },
        {
            "title": "Machine Learning Basics",
            "content": "Machine learning is a method of data analysis that automates analytical model building."
        },
        {
            "title": "機器學習基礎",
            "content": "機器學習是一種數據分析方法，它能自動建立分析模型。通過學習數據模式來做出預測。"
        },
    ]

    print("Document Collection Analysis:\n")

    stats = {"en": 0, "zh": 0, "total_chunks": 0}

    for doc in documents:
        lang = detect_language(doc["content"])
        chunk_size = 100 if lang == "zh" else 200

        chunks = split_chinese_text(
            doc["content"],
            chunk_size=chunk_size,
            chunk_overlap=20
        )

        stats[lang] += 1
        stats["total_chunks"] += len(chunks)

        print(f"Title: {doc['title']}")
        print(f"  Language: {lang}")
        print(f"  Content length: {len(doc['content'])} chars")
        print(f"  Chunks created: {len(chunks)}")
        print(f"  Chunk size used: {chunk_size}")
        print()

    print("Collection Statistics:")
    print(f"  English documents: {stats['en']}")
    print(f"  Chinese documents: {stats['zh']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Avg chunks per doc: {stats['total_chunks'] / len(documents):.1f}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("   Traditional Chinese Support Examples")
    print("   TextReadingRAG System")
    print("="*60)

    try:
        # Run examples
        example_language_detection()
        example_chinese_tokenization()
        example_chinese_text_splitting()
        example_multilingual_comparison()
        example_tokenizer_comparison()
        example_real_world_scenario()

        print("\n" + "="*60)
        print("   All examples completed successfully!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        raise


if __name__ == "__main__":
    main()
