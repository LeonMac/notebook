import os
import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.similarities import WmdSimilarity
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram, linkage
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

glove_file = os.path.join(api.BASE_DIR, 'glove.2024.wikigiga.50d/wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt')
word2vec_file = os.path.join(api.BASE_DIR,'converted_word2vec.txt')

from gensim.scripts.glove2word2vec import glove2word2vec
# glove2word2vec(glove_file, word2vec_file) # run just once time

class WordSimilarityAnalyzer:
    """
    使用 gensim 的词相似度分析器
    """
    
    def __init__(self, model_name: str = 'glove-2024-wiki-gigaword-50'):
        """
        初始化分析器
        
        Args:
            model_name: 预训练模型名称
            force_download: 是否强制重新下载模型
        """
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.model_name = model_name
        self.model = None
        self.vocab = set()
        self._load_model()
            
    def _load_model(self):
        """加载预训练词向量模型"""
        try:
            self.logger.info(f"加载模型： {self.model_name} 这可能需要花点时间！")
            self.model = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
            
            # 构建词汇表
            self.vocab = set(self.model.key_to_index.keys())
            self.logger.info(f"模型加载成功: {self.model_name}")
            self.logger.info(f"词汇表大小: {len(self.vocab)}")
            self.logger.info(f"向量维度: {self.model.vector_size}")
            # self.logger.info(f"向量数据精度: {self.model.vector_size}")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def word_similarity(self, word1: str, word2: str) -> float:
        """
        计算两个词的余弦相似度
        
        Args:
            word1: 第一个词
            word2: 第二个词
            
        Returns:
            相似度分数 (0-1)
        """
        if not self._validate_words([word1, word2]):
            return 0.0
        
        try:
            return self.model.similarity(word1, word2)
        except Exception as e:
            self.logger.warning(f"计算相似度失败: {e}")
            return 0.0
    
    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        查找与目标词最相似的词
        
        Args:
            word: 目标词
            topn: 返回结果数量
            
        Returns:
            [(相似词, 相似度), ...]
        """
        if word not in self.vocab:
            self.logger.warning(f"词 '{word}' 不在词汇表中")
            return []
        
        try:
            return self.model.most_similar(word, topn=topn)
        except Exception as e:
            self.logger.warning(f"查找相似词失败: {e}")
            return []
    
    def analogy(self, positive: List[str], negative: List[str], topn: int = 10) -> List[Tuple[str, float]]:
        """
        词类比推理 (如: king - man + woman = queen)
        
        Args:
            positive: 正向词列表
            negative: 负向词列表
            topn: 返回结果数量
            
        Returns:
            [(结果词, 相似度), ...]
        """
        words = positive + negative
        if not self._validate_words(words):
            return []
        
        try:
            return self.model.most_similar(positive=positive, negative=negative, topn=topn)
        except Exception as e:
            self.logger.warning(f"词类比推理失败: {e}")
            return []
    
    def sentence_similarity(self, sentence1: str, sentence2: str, method: str = 'avg') -> float:
        """
        计算两个句子的相似度
        
        Args:
            sentence1: 第一个句子
            sentence2: 第二个句子
            method: 计算方法 ('avg', 'wmd')
            
        Returns:
            句子相似度分数
        """
        words1 = self._preprocess_sentence(sentence1)
        words2 = self._preprocess_sentence(sentence2)
        
        if not words1 or not words2:
            return 0.0
        
        if method == 'wmd':
            return self._wmd_similarity(words1, words2)
        else:
            return self._avg_vector_similarity(words1, words2)
    
    def _avg_vector_similarity(self, words1: List[str], words2: List[str]) -> float:
        """基于平均向量的句子相似度计算"""
        vec1 = self._get_sentence_vector(words1)
        vec2 = self._get_sentence_vector(words2)
        
        if vec1 is None or vec2 is None:
            return 0.0
        
        # 计算余弦相似度
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _wmd_similarity(self, words1: List[str], words2: List[str]) -> float:
        """基于词移距离的句子相似度计算"""
        try:
            # 使用 gensim 的 WMD 实现
            distance = self.model.wmdistance(words1, words2)
            # 将距离转换为相似度 (距离越小，相似度越大)
            return 1 / (1 + distance)
        except Exception as e:
            self.logger.warning(f"WMD计算失败: {e}")
            return 0.0
    
    def _get_sentence_vector(self, words: List[str]) -> Optional[np.ndarray]:
        """获取句子的平均向量"""
        valid_vectors = []
        for word in words:
            if word in self.vocab:
                valid_vectors.append(self.model[word])
        
        if not valid_vectors:
            return None
        
        return np.mean(valid_vectors, axis=0)
    
    def _preprocess_sentence(self, sentence: str) -> List[str]:
        """预处理句子"""
        # 简单的分词和清洗
        words = sentence.lower().split()
        return [word.strip('.,!?;:"') for word in words if word.strip('.,!?;:"')]
    
    def _validate_words(self, words: List[str]) -> bool:
        """验证词是否在词汇表中"""
        for word in words:
            if word not in self.vocab:
                self.logger.warning(f"词 '{word}' 不在词汇表中")
                return False
        return True
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """获取词的向量表示"""
        if word in self.vocab:
            return self.model[word]
        return None
    
    def vocabulary_coverage(self, words: List[str]) -> float:
        """
        计算词汇表覆盖率
        
        Args:
            words: 词列表
            
        Returns:
            覆盖率 (0-1)
        """
        valid_count = sum(1 for word in words if word in self.vocab)
        return valid_count / len(words) if words else 0.0
    
    def batch_similarity(self, word_pairs: List[Tuple[str, str]]) -> List[float]:
        """
        批量计算词对相似度
        
        Args:
            word_pairs: 词对列表
            
        Returns:
            相似度列表
        """
        similarities = []
        for word1, word2 in word_pairs:
            similarities.append(self.word_similarity(word1, word2))
        return similarities
    
    def export_similarity_report(self, target_words: List[str], topn: int = 5) -> pd.DataFrame:
        """
        导出相似词报告
        
        Args:
            target_words: 目标词列表
            topn: 每个词的相似词数量
            
        Returns:
            DataFrame 报告
        """
        report_data = []
        
        for word in target_words:
            similar_words = self.most_similar(word, topn=topn)
            for similar_word, similarity in similar_words:
                report_data.append({
                    'target_word': word,
                    'similar_word': similar_word,
                    'similarity': similarity
                })
        
        return pd.DataFrame(report_data)
    
class AdvancedWordSimilarityAnalyzer(WordSimilarityAnalyzer):
    """扩展的词相似度分析器"""
    
    def __init__(self, model_name: str = 'glove-wiki-gigaword-50'):
        super().__init__(model_name)
    
    def semantic_relationship_strength(self, word_pairs: List[Tuple[str, str]]) -> float:
        """
        计算语义关系强度
        
        Args:
            word_pairs: 具有相同语义关系的词对列表
            (如: [('king', 'queen'), ('man', 'woman'), ('boy', 'girl')])
            
        Returns:
            关系强度分数
        """
        if len(word_pairs) < 2:
            return 0.0
        
        similarities = []
        for word1, word2 in word_pairs:
            if word1 in self.vocab and word2 in self.vocab:
                similarities.append(self.word_similarity(word1, word2))
        
        return np.mean(similarities) if similarities else 0.0
    
    def find_semantic_axis(self, positive_examples: List[str], negative_examples: List[str]) -> np.ndarray:
        """
        寻找语义轴 (如: 性别轴、时态轴)
        
        Args:
            positive_examples: 正向示例词
            negative_examples: 负向示例词
            
        Returns:
            语义轴向量
        """
        positive_vectors = []
        negative_vectors = []
        
        for word in positive_examples:
            if word in self.vocab:
                positive_vectors.append(self.model[word])
        
        for word in negative_examples:
            if word in self.vocab:
                negative_vectors.append(self.model[word])
        
        if not positive_vectors or not negative_vectors:
            return None
        
        avg_positive = np.mean(positive_vectors, axis=0)
        avg_negative = np.mean(negative_vectors, axis=0)
        
        return avg_positive - avg_negative
    
    def project_on_axis(self, word: str, axis: np.ndarray) -> float:
        """
        将词投影到语义轴上
        
        Args:
            word: 目标词
            axis: 语义轴向量
            
        Returns:
            投影分数
        """
        if word not in self.vocab:
            return 0.0
        
        word_vector = self.model[word]
        projection = np.dot(word_vector, axis) / np.linalg.norm(axis)
        return projection





def word_analogy_analysis(analyzer, word_pair, single_word, topn=3, description=None):
    """
    执行词类比推理分析
    
    Args:
        analyzer: WordSimilarityAnalyzer实例
        word_pair: 词对，如 ['king', 'man']
        single_word: 单个词，如 'woman'
        topn: 返回最相似词的数量
        description: 可选的描述文本
    
    Returns:
        tuple: (结果列表, 是否成功)
    """
    if description is None:
        description = f"{word_pair[0]} - {word_pair[1]} + {single_word}"
    
    print(f"\n词类比推理 ({description}):")
    
    try:
        # 执行词类比推理
        analogy_results = analyzer.analogy(
            positive=[word_pair[0], single_word], 
            negative=[word_pair[1]], 
            topn=topn
        )
        
        if not analogy_results:
            print("  未找到合适的类比结果")
            return [], False
        
        # 输出结果
        for word, sim in analogy_results:
            print(f"  {word}: {sim:.4f}")
        
        return analogy_results, True
        
    except Exception as e:
        print(f"  词类比推理失败: {e}")
        return [], False


def batch_word_analogy(analyzer, analogy_tests):
    """
    批量执行多个词类比推理测试
    
    Args:
        analyzer: WordSimilarityAnalyzer实例
        analogy_tests: 测试列表，每个元素为 (word_pair, single_word, description)
    
    Returns:
        dict: 所有测试结果
    """
    results = {}
    
    print("=" * 60)
    print("批量词类比推理分析")
    print("=" * 60)
    
    for i, test in enumerate(analogy_tests, 1):
        if len(test) == 2:
            word_pair, single_word = test
            description = f"测试 {i}: {word_pair[0]} - {word_pair[1]} + {single_word}"
        else:
            word_pair, single_word, description = test
        
        print(f"\n[{i}/{len(analogy_tests)}] {description}")
        
        analogy_results, success = word_analogy_analysis(
            analyzer, word_pair, single_word, topn=3, description=description
        )
        
        results[description] = {
            'results': analogy_results,
            'success': success,
            'word_pair': word_pair,
            'single_word': single_word
        }
    
    return results

def analogy_strength_analysis(analyzer, base_pair, test_words):
    """
    分析词类比关系的强度（修正版）
    
    Args:
        analyzer: WordSimilarityAnalyzer实例
        base_pair: 基础词对，如 ['man', 'woman']
        test_words: 测试词列表，如 ['king', 'queen', 'boy', 'girl', 'prince', 'princess']
    
    Returns:
        dict: 强度分析结果
    """
    print(f"\n词类比关系强度分析: {base_pair[0]} - {base_pair[1]} 关系")
    print("-" * 50)
    
    results = {}
    
    for i in range(0, len(test_words), 2):
        if i + 1 < len(test_words):
            word1 = test_words[i]      # 如 'king'
            expected_word = test_words[i + 1]  # 如 'queen'
            
            # 正确的类比推理：word1 - base_pair[0] + base_pair[1]
            description = f"{word1} - {base_pair[0]} + {base_pair[1]}"
            
            print(f"\n测试: {word1} → {expected_word}")
            
            # 执行正确的类比推理
            analogy_results, success = word_analogy_analysis(
                analyzer, 
                word_pair=[word1, base_pair[0]],  # ['king', 'man']
                single_word=base_pair[1],         # 'woman'
                topn=5, 
                description=description
            )
            
            # 检查预期结果是否在top结果中
            found_rank = None
            found_similarity = 0
            
            for rank, (word, similarity) in enumerate(analogy_results, 1):
                if word == expected_word:
                    found_rank = rank
                    found_similarity = similarity
                    break
            
            results[f"{word1}→{expected_word}"] = {
                'expected': expected_word,
                'found_rank': found_rank,
                'found_similarity': found_similarity,
                'all_results': analogy_results,
                'success': found_rank is not None
            }
            
            if found_rank:
                print(f"  ✅ 预期词 '{expected_word}' 出现在第 {found_rank} 位 (相似度: {found_similarity:.4f})")
            else:
                print(f"  ❌ 预期词 '{expected_word}' 未出现在前 {len(analogy_results)} 个结果中")
                # 显示实际的前5个结果
                print(f"  实际结果: {[word for word, sim in analogy_results]}")
    
    return results

def calculate_relation_strength(analyzer, relation_pairs, relation_name="关系"):
    """
    计算语义关系的平均强度
    
    Args:
        analyzer: WordSimilarityAnalyzer实例
        relation_pairs: 关系词对列表，如 [('king','queen'), ('man','woman'), ...]
        relation_name: 关系名称
    """
    print(f"\n{relation_name}强度统计")
    print("-" * 40)
    
    success_count = 0
    total_rank = 0
    total_similarity = 0
    successful_pairs = []
    
    for word1, word2 in relation_pairs:
        # 测试该关系对
        base_relation = [word1, word2]
        test_words = [word1, word2]  # 测试自身
        
        results = analogy_strength_analysis(analyzer, base_relation, test_words)
        
        # 分析结果
        key = f"{word1}→{word2}"
        if key in results and results[key]['success']:
            success_count += 1
            total_rank += results[key]['found_rank']
            total_similarity += results[key]['found_similarity']
            successful_pairs.append((word1, word2, results[key]['found_rank'], results[key]['found_similarity']))
    
    # 输出统计结果
    if success_count > 0:
        avg_rank = total_rank / success_count
        avg_similarity = total_similarity / success_count
        success_rate = success_count / len(relation_pairs)
        
        print(f"\n📊 {relation_name}强度统计结果:")
        print(f"  测试词对数量: {len(relation_pairs)}")
        print(f"  成功识别数量: {success_count}")
        print(f"  识别成功率: {success_rate:.2%}")
        print(f"  平均排名: {avg_rank:.2f}")
        print(f"  平均相似度: {avg_similarity:.4f}")
        
        print(f"\n成功识别的词对:")
        for word1, word2, rank, sim in successful_pairs:
            print(f"  {word1} → {word2}: 排名{rank}, 相似度{sim:.4f}")
    else:
        print("没有成功识别的词对")

def semantic_relationship_test(analyzer, relationship_type="all"):
    """
    测试常见的语义关系
    
    Args:
        analyzer: WordSimilarityAnalyzer实例
        relationship_type: 关系类型 ('all', 'gender', 'plural', 'country-capital', 'verb-tense')
    
    Returns:
        dict: 测试结果
    """
    # 定义常见的语义关系测试用例
    test_cases = {
        'gender': [
            (['king', 'man'], 'woman', "性别关系: king - man + woman"),
            (['actor', 'man'], 'woman', "性别关系: actor - man + woman"),
            (['prince', 'man'], 'woman', "性别关系: prince - man + woman"),
        ],
        'plural': [
            (['dogs', 'dog'], 'cat', "复数关系: dogs - dog + cat"),
            (['children', 'child'], 'adult', "复数关系: children - child + adult"),
            (['mice', 'mouse'], 'rat', "复数关系: mice - mouse + rat"),
        ],
        'country-capital': [
            (['paris', 'france'], 'germany', "国家-首都: paris - france + germany"),
            (['london', 'england'], 'france', "国家-首都: london - england + france"),
            (['tokyo', 'japan'], 'china', "国家-首都: tokyo - japan + china"),
        ],
        'verb-tense': [
            (['running', 'run'], 'walk', "动词时态: running - run + walk"),
            (['swam', 'swim'], 'drink', "动词时态: swam - swim + drink"),
            (['thought', 'think'], 'know', "动词时态: thought - think + know"),
        ]
    }
    
    if relationship_type == 'all':
        selected_tests = []
        for tests in test_cases.values():
            selected_tests.extend(tests)
    else:
        selected_tests = test_cases.get(relationship_type, [])
    
    print(f"\n语义关系测试: {relationship_type}")
    print("=" * 60)
    
    return batch_word_analogy(analyzer, selected_tests)



def pairwise_similarity_analysis(analyzer, word_list):
    """
    对单词列表进行两两相似度分析，生成详细报告
    
    Args:
        analyzer: WordSimilarityAnalyzer实例
        word_list: 单词列表
    
    Returns:
        dict: 包含各种分析结果的字典
    """
    # 过滤掉不在词汇表中的单词
    valid_words = [word for word in word_list if word in analyzer.vocab]
    invalid_words = [word for word in word_list if word not in analyzer.vocab]
    
    print(f"有效单词数: {len(valid_words)}/{len(word_list)}")
    if invalid_words:
        print(f"无效单词(不在词汇表中): {invalid_words}")
    
    if len(valid_words) < 2:
        print("有效单词数量不足，无法进行分析")
        return None
    
    # 生成所有单词对组合
    word_pairs = list(combinations(valid_words, 2))
    
    # 批量计算相似度
    similarities = analyzer.batch_similarity(word_pairs)
    
    # 创建相似度矩阵
    similarity_matrix = create_similarity_matrix(valid_words, word_pairs, similarities)
    
    # 生成详细报告
    report = generate_detailed_report(valid_words, word_pairs, similarities, similarity_matrix)
    
    # 可视化相似度矩阵
    visualize_similarity_matrix(similarity_matrix, valid_words)
    
    # 识别高相似度和低相似度词对
    identify_extreme_pairs(word_pairs, similarities)
    
    return report

def create_similarity_matrix(words, word_pairs, similarities):
    """创建相似度矩阵"""
    n = len(words)
    matrix = np.identity(n)  # 对角线为1
    
    # 填充相似度矩阵
    word_to_idx = {word: idx for idx, word in enumerate(words)}
    for (word1, word2), similarity in zip(word_pairs, similarities):
        i, j = word_to_idx[word1], word_to_idx[word2]
        matrix[i, j] = similarity
        matrix[j, i] = similarity
    
    return matrix

def generate_detailed_report(words, word_pairs, similarities, similarity_matrix):
    """生成详细分析报告"""
    # 创建词对相似度表格
    pair_data = []
    for (word1, word2), similarity in zip(word_pairs, similarities):
        pair_data.append({
            'word1': word1,
            'word2': word2,
            'similarity': similarity,
            'similarity_category': categorize_similarity(similarity)
        })
    
    pair_df = pd.DataFrame(pair_data).sort_values('similarity', ascending=False)
    
    # 计算每个单词的平均相似度
    word_stats = []
    word_to_idx = {word: idx for idx, word in enumerate(words)}
    
    for word in words:
        idx = word_to_idx[word]
        # 排除与自身的相似度(1.0)
        other_similarities = [similarity_matrix[idx, j] for j in range(len(words)) if j != idx]
        avg_similarity = np.mean(other_similarities) if other_similarities else 0
        
        word_stats.append({
            'word': word,
            'avg_similarity': avg_similarity,
            'max_similarity': max(other_similarities) if other_similarities else 0,
            'min_similarity': min(other_similarities) if other_similarities else 0
        })
    
    word_stats_df = pd.DataFrame(word_stats).sort_values('avg_similarity', ascending=False)
    
    # 整体统计
    overall_stats = {
        'total_word_pairs': len(word_pairs),
        'mean_similarity': np.mean(similarities),
        'median_similarity': np.median(similarities),
        'std_similarity': np.std(similarities),
        'max_similarity': np.max(similarities),
        'min_similarity': np.min(similarities)
    }
    
    # 打印报告
    print("=" * 60)
    print("单词两两相似度分析报告")
    print("=" * 60)
    
    print(f"\n整体统计:")
    for stat, value in overall_stats.items():
        print(f"  {stat}: {value:.4f}")
    
    print(f"\n单词相似度排名 (按平均相似度):")
    for i, row in word_stats_df.iterrows():
        print(f"  {row['word']}: 平均{row['avg_similarity']:.4f} "
              f"(最高{row['max_similarity']:.4f}, 最低{row['min_similarity']:.4f})")
    
    print(f"\nTop 10 最相似词对:")
    for i, row in pair_df.head(10).iterrows():
        print(f"  {row['word1']} - {row['word2']}: {row['similarity']:.4f} [{row['similarity_category']}]")
    
    print(f"\nTop 10 最不相似词对:")
    for i, row in pair_df.tail(10).iterrows():
        print(f"  {row['word1']} - {row['word2']}: {row['similarity']:.4f} [{row['similarity_category']}]")
    
    return {
        'pairwise_data': pair_df,
        'word_stats': word_stats_df,
        'similarity_matrix': similarity_matrix,
        'overall_stats': overall_stats,
        'valid_words': words
    }

def categorize_similarity(similarity):
    """根据相似度值分类"""
    if similarity >= 0.7:
        return "极高相似度"
    elif similarity >= 0.5:
        return "高相似度"
    elif similarity >= 0.3:
        return "中等相似度"
    elif similarity >= 0.1:
        return "低相似度"
    else:
        return "极低相似度"

def visualize_similarity_matrix(matrix, words):
    """可视化相似度矩阵"""
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(matrix, dtype=bool))  # 创建上三角掩码
    
    sns.heatmap(matrix, 
                mask=mask,
                xticklabels=words, 
                yticklabels=words,
                cmap='RdYlBu_r', 
                annot=True, 
                fmt='.2f',
                center=0.5,
                square=True)
    
    plt.title('单词相似度矩阵', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def identify_extreme_pairs(word_pairs, similarities, threshold_high=0.6, threshold_low=0.2):
    """识别极高和极低相似度的词对"""
    high_sim_pairs = [(pair, sim) for pair, sim in zip(word_pairs, similarities) if sim >= threshold_high]
    low_sim_pairs = [(pair, sim) for pair, sim in zip(word_pairs, similarities) if sim <= threshold_low]
    
    print(f"\n极高相似度词对 (≥{threshold_high}):")
    for pair, sim in sorted(high_sim_pairs, key=lambda x: x[1], reverse=True):
        print(f"  {pair[0]} - {pair[1]}: {sim:.4f}")
    
    print(f"\n极低相似度词对 (≤{threshold_low}):")
    for pair, sim in sorted(low_sim_pairs, key=lambda x: x[1]):
        print(f"  {pair[0]} - {pair[1]}: {sim:.4f}")

def cluster_words_by_similarity(analyzer, word_list, n_clusters=None, method='hierarchical', **kwargs):
    """
    基于词向量相似度对单词进行聚类
    
    Args:
        analyzer: WordSimilarityAnalyzer实例
        word_list: 单词列表
        n_clusters: 聚类数量 (None表示自动确定)
        method: 聚类方法 ('hierarchical', 'dbscan')
        **kwargs: 其他聚类参数
    
    Returns:
        dict: 聚类结果
    """
    # 过滤有效单词并获取词向量
    valid_words = [word for word in word_list if word in analyzer.vocab]
    word_vectors = [analyzer.get_word_vector(word) for word in valid_words]
    
    print(f"用于聚类的有效单词: {len(valid_words)}/{len(word_list)}")
    
    if len(valid_words) < 3:
        print("有效单词数量不足，无法进行聚类")
        return None
    
    # 创建距离矩阵 (1 - 相似度)
    distance_matrix = create_distance_matrix(valid_words, analyzer)
    
    # 执行聚类
    if method == 'hierarchical':
        clustering_result = hierarchical_clustering(valid_words, distance_matrix, n_clusters)
    elif method == 'dbscan':
        # 从kwargs中获取DBSCAN参数，或使用默认值
        eps = kwargs.get('eps', 0.7)
        min_samples = kwargs.get('min_samples', 2)
        clustering_result = dbscan_clustering(valid_words, distance_matrix, eps, min_samples)
    else:
        raise ValueError("不支持的聚类方法")
    
    # 可视化聚类结果
    visualize_clustering_results(clustering_result, valid_words, word_vectors)
    
    return clustering_result

def create_distance_matrix(words, analyzer):
    """创建距离矩阵 (1 - 相似度)"""
    n = len(words)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                distance_matrix[i, j] = 0
            else:
                similarity = analyzer.word_similarity(words[i], words[j])
                distance_matrix[i, j] = 1 - similarity  # 转换为距离
    
    return distance_matrix

def hierarchical_clustering(words, distance_matrix, n_clusters=None):
    """层次聚类"""
    if n_clusters is None:
        # 自动确定聚类数量 - 使用肘部法则的简化版本
        n_clusters = suggest_optimal_clusters(distance_matrix)
        print(f"自动确定的聚类数量: {n_clusters}")
    
    # 执行层次聚类
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    
    labels = clustering.fit_predict(distance_matrix)
    
    # 组织聚类结果
    clusters = {}
    for word, label in zip(words, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(word)
    
    # 计算聚类内平均相似度
    cluster_stats = calculate_cluster_stats(clusters, distance_matrix, words)
    
    result = {
        'method': 'hierarchical',
        'clusters': clusters,
        'labels': labels,
        'n_clusters': n_clusters,
        'cluster_stats': cluster_stats
    }
    
    print_clustering_results(result, words)
    plot_dendrogram(distance_matrix, words)
    
    return result

def dbscan_clustering(words, distance_matrix, eps=0.7, min_samples=2):
    """DBSCAN聚类"""
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(distance_matrix)
    
    # 组织聚类结果 (包括噪声点)
    clusters = {}
    noise_points = []
    
    for word, label in zip(words, labels):
        if label == -1:  # 噪声点
            noise_points.append(word)
        else:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(word)
    
    # 计算聚类统计
    cluster_stats = calculate_cluster_stats(clusters, distance_matrix, words)
    
    result = {
        'method': 'dbscan',
        'clusters': clusters,
        'noise_points': noise_points,
        'labels': labels,
        'n_clusters': len(clusters),
        'cluster_stats': cluster_stats
    }
    
    print_clustering_results(result, words)
    
    return result

def suggest_optimal_clusters(distance_matrix, max_clusters=8):
    """建议最优聚类数量 (简化版肘部法则)"""
    distortions = []
    max_possible = min(len(distance_matrix) - 1, max_clusters)
    
    for k in range(1, max_possible + 1):
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(distance_matrix)
        
        # 计算类内平均距离
        distortion = calculate_within_cluster_distance(distance_matrix, labels, k)
        distortions.append(distortion)
    
    # 简单的肘部法则：选择拐点
    if len(distortions) > 1:
        # 计算二阶差分找到最大变化点
        second_diff = np.diff(distortions, 2)
        if len(second_diff) > 0:
            optimal_k = np.argmax(second_diff) + 2  # +2 因为二阶差分
            return min(optimal_k, max_possible)
    
    return min(3, max_possible)  # 默认返回3

def calculate_within_cluster_distance(distance_matrix, labels, n_clusters):
    """计算类内平均距离"""
    total_distance = 0
    count = 0
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) > 1:
            # 计算类内所有点对的距离
            for i in range(len(cluster_indices)):
                for j in range(i + 1, len(cluster_indices)):
                    total_distance += distance_matrix[cluster_indices[i], cluster_indices[j]]
                    count += 1
    
    return total_distance / count if count > 0 else 0

def calculate_cluster_stats(clusters, distance_matrix, words):
    """计算聚类统计信息"""
    word_to_idx = {word: idx for idx, word in enumerate(words)}
    stats = {}
    
    for cluster_id, cluster_words in clusters.items():
        if len(cluster_words) < 2:
            continue
            
        # 计算类内平均距离
        cluster_indices = [word_to_idx[word] for word in cluster_words]
        within_distances = []
        
        for i in range(len(cluster_indices)):
            for j in range(i + 1, len(cluster_indices)):
                within_distances.append(distance_matrix[cluster_indices[i], cluster_indices[j]])
        
        avg_within_distance = np.mean(within_distances) if within_distances else 0
        avg_within_similarity = 1 - avg_within_distance  # 转换回相似度
        
        stats[cluster_id] = {
            'size': len(cluster_words),
            'avg_within_similarity': avg_within_similarity,
            'words': cluster_words
        }
    
    return stats

def print_clustering_results(result, words):
    """打印聚类结果"""
    print("\n" + "=" * 50)
    print("聚类分析结果")
    print("=" * 50)
    
    print(f"聚类方法: {result['method']}")
    print(f"聚类数量: {result['n_clusters']}")
    
    if 'noise_points' in result and result['noise_points']:
        print(f"噪声点: {result['noise_points']}")
    
    print("\n各聚类详情:")
    for cluster_id, stats in result['cluster_stats'].items():
        print(f"\n聚类 {cluster_id}:")
        print(f"  单词数量: {stats['size']}")
        print(f"  类内平均相似度: {stats['avg_within_similarity']:.4f}")
        print(f"  包含单词: {', '.join(stats['words'])}")

def plot_dendrogram(distance_matrix, words):
    """绘制树状图"""
    plt.figure(figsize=(12, 8))
    
    # 计算链接矩阵
    linked = linkage(distance_matrix, 'average')
    
    # 绘制树状图
    dendrogram(linked,
               orientation='top',
               labels=words,
               distance_sort='descending',
               show_leaf_counts=True)
    
    plt.title('单词聚类树状图', fontsize=16)
    plt.xlabel('单词')
    plt.ylabel('距离')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def visualize_clustering_results(clustering_result, words, word_vectors):
    """可视化聚类结果"""
    # 使用多维缩放进行降维可视化
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    
    # 创建距离矩阵用于MDS
    n = len(words)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                distance_matrix[i, j] = 0
            else:
                vec1, vec2 = word_vectors[i], word_vectors[j]
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                distance_matrix[i, j] = 1 - similarity
    
    # 执行MDS降维
    coords = mds.fit_transform(distance_matrix)
    
    # 绘制散点图
    plt.figure(figsize=(12, 8))
    
    labels = clustering_result['labels']
    unique_labels = set(labels)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:  # 噪声点
            cluster_coords = coords[labels == label]
            plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], 
                       c='gray', s=100, alpha=0.6, label='噪声点')
        else:
            cluster_coords = coords[labels == label]
            plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], 
                       c=[color], s=100, alpha=0.7, label=f'聚类 {label}')
    
    # 添加单词标签
    for i, word in enumerate(words):
        plt.annotate(word, (coords[i, 0], coords[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.title('单词聚类可视化', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def word_analogy_analysis(analyzer, word_pair, single_word, topn=3, description=None):
    """
    执行词类比推理分析
    
    Args:
        analyzer: WordSimilarityAnalyzer实例
        word_pair: 词对，如 ['king', 'man']
        single_word: 单个词，如 'woman'
        topn: 返回最相似词的数量
        description: 可选的描述文本
    
    Returns:
        tuple: (结果列表, 是否成功)
    """
    if description is None:
        description = f"{word_pair[0]} - {word_pair[1]} + {single_word}"
    
    print(f"\n词类比推理 ({description}):")
    
    try:
        # 执行词类比推理
        analogy_results = analyzer.analogy(
            positive=[word_pair[0], single_word], 
            negative=[word_pair[1]], 
            topn=topn
        )
        
        if not analogy_results:
            print("  未找到合适的类比结果")
            return [], False
        
        # 输出结果
        for word, sim in analogy_results:
            print(f"  {word}: {sim:.4f}")
        
        return analogy_results, True
        
    except Exception as e:
        print(f"  词类比推理失败: {e}")
        return [], False

def batch_word_analogy(analyzer, analogy_tests):
    """
    批量执行多个词类比推理测试
    
    Args:
        analyzer: WordSimilarityAnalyzer实例
        analogy_tests: 测试列表，每个元素为 (word_pair, single_word, description)
    
    Returns:
        dict: 所有测试结果
    """
    results = {}
    
    print("=" * 60)
    print("批量词类比推理分析")
    print("=" * 60)
    
    for i, test in enumerate(analogy_tests, 1):
        if len(test) == 2:
            word_pair, single_word = test
            description = f"测试 {i}: {word_pair[0]} - {word_pair[1]} + {single_word}"
        else:
            word_pair, single_word, description = test
        
        print(f"\n[{i}/{len(analogy_tests)}] {description}")
        
        analogy_results, success = word_analogy_analysis(
            analyzer, word_pair, single_word, topn=3, description=description
        )
        
        results[description] = {
            'results': analogy_results,
            'success': success,
            'word_pair': word_pair,
            'single_word': single_word
        }
    
    return results

def analogy_strength_analysis(analyzer, base_pair, test_words):
    """
    分析词类比关系的强度（修正版）
    
    Args:
        analyzer: WordSimilarityAnalyzer实例
        base_pair: 基础词对，如 ['man', 'woman']
        test_words: 测试词列表，如 ['king', 'queen', 'boy', 'girl', 'prince', 'princess']
    
    Returns:
        dict: 强度分析结果
    """
    print(f"\n词类比关系强度分析: {base_pair[0]} - {base_pair[1]} 关系")
    print("-" * 50)
    
    results = {}
    
    for i in range(0, len(test_words), 2):
        if i + 1 < len(test_words):
            word1 = test_words[i]      # 如 'king'
            expected_word = test_words[i + 1]  # 如 'queen'
            
            # 正确的类比推理：word1 - base_pair[0] + base_pair[1]
            description = f"{word1} - {base_pair[0]} + {base_pair[1]}"
            
            print(f"\n测试: {word1} → {expected_word}")
            
            # 执行正确的类比推理
            analogy_results, success = word_analogy_analysis(
                analyzer, 
                word_pair=[word1, base_pair[0]],  # ['king', 'man']
                single_word=base_pair[1],         # 'woman'
                topn=5, 
                description=description
            )
            
            # 检查预期结果是否在top结果中
            found_rank = None
            found_similarity = 0
            
            for rank, (word, similarity) in enumerate(analogy_results, 1):
                if word == expected_word:
                    found_rank = rank
                    found_similarity = similarity
                    break
            
            results[f"{word1}→{expected_word}"] = {
                'expected': expected_word,
                'found_rank': found_rank,
                'found_similarity': found_similarity,
                'all_results': analogy_results,
                'success': found_rank is not None
            }
            
            if found_rank:
                print(f"  ✅ 预期词 '{expected_word}' 出现在第 {found_rank} 位 (相似度: {found_similarity:.4f})")
            else:
                print(f"  ❌ 预期词 '{expected_word}' 未出现在前 {len(analogy_results)} 个结果中")
                # 显示实际的前5个结果
                print(f"  实际结果: {[word for word, sim in analogy_results]}")
    
    return results

def semantic_relationship_test(analyzer, relationship_type="all"):
    """
    测试常见的语义关系
    
    Args:
        analyzer: WordSimilarityAnalyzer实例
        relationship_type: 关系类型 ('all', 'gender', 'plural', 'country-capital', 'verb-tense')
    
    Returns:
        dict: 测试结果
    """
    # 定义常见的语义关系测试用例
    test_cases = {
        'gender': [
            (['king', 'man'], 'woman', "性别关系: king - man + woman"),
            (['actor', 'man'], 'woman', "性别关系: actor - man + woman"),
            (['prince', 'man'], 'woman', "性别关系: prince - man + woman"),
        ],
        'plural': [
            (['dogs', 'dog'], 'cat', "复数关系: dogs - dog + cat"),
            (['children', 'child'], 'adult', "复数关系: children - child + adult"),
            (['mice', 'mouse'], 'rat', "复数关系: mice - mouse + rat"),
        ],
        'country-capital': [
            (['paris', 'france'], 'germany', "国家-首都: paris - france + germany"),
            (['london', 'england'], 'france', "国家-首都: london - england + france"),
            (['tokyo', 'japan'], 'china', "国家-首都: tokyo - japan + china"),
        ],
        'verb-tense': [
            (['running', 'run'], 'walk', "动词时态: running - run + walk"),
            (['swam', 'swim'], 'drink', "动词时态: swam - swim + drink"),
            (['thought', 'think'], 'know', "动词时态: thought - think + know"),
        ]
    }
    
    if relationship_type == 'all':
        selected_tests = []
        for tests in test_cases.values():
            selected_tests.extend(tests)
    else:
        selected_tests = test_cases.get(relationship_type, [])
    
    print(f"\n语义关系测试: {relationship_type}")
    print("=" * 60)
    
    return batch_word_analogy(analyzer, selected_tests)

def analogy_confidence_analysis(analyzer, test_cases, topn=5):
    """
    分析词类比推理的置信度
    
    Args:
        analyzer: WordSimilarityAnalyzer实例
        test_cases: 测试用例列表，每个元素为 (word_pair, single_word, expected)
        topn: 考虑的前N个结果
    
    Returns:
        dict: 置信度分析结果
    """
    print("词类比推理置信度分析")
    print("=" * 50)
    
    results = {
        'total_tests': len(test_cases),
        'correct_top1': 0,
        'correct_top3': 0,
        'correct_top5': 0,
        'avg_rank': 0,
        'avg_similarity': 0,
        'detailed_results': []
    }
    
    ranks = []
    similarities = []
    
    for i, (word_pair, single_word, expected) in enumerate(test_cases, 1):
        description = f"{word_pair[0]} - {word_pair[1]} + {single_word}"
        print(f"\n[{i}/{len(test_cases)}] {description} (期望: {expected})")
        
        analogy_results, success = word_analogy_analysis(
            analyzer, word_pair, single_word, topn=topn
        )
        
        # 查找预期结果的排名和相似度
        found_rank = None
        found_similarity = 0
        
        for rank, (word, similarity) in enumerate(analogy_results, 1):
            if word == expected:
                found_rank = rank
                found_similarity = similarity
                break
        
        # 更新统计
        if found_rank == 1:
            results['correct_top1'] += 1
        if found_rank and found_rank <= 3:
            results['correct_top3'] += 1
        if found_rank and found_rank <= 5:
            results['correct_top5'] += 1
        
        if found_rank:
            ranks.append(found_rank)
            similarities.append(found_similarity)
            print(f"  ✅ 预期词 '{expected}' 排名: {found_rank}, 相似度: {found_similarity:.4f}")
        else:
            ranks.append(topn + 1)  # 表示不在前topn中
            similarities.append(0)
            print(f"  ❌ 预期词 '{expected}' 未出现在前 {topn} 个结果中")
        
        results['detailed_results'].append({
            'test_case': description,
            'expected': expected,
            'found_rank': found_rank,
            'found_similarity': found_similarity,
            'all_results': analogy_results
        })
    
    # 计算平均值
    if ranks:
        results['avg_rank'] = sum(ranks) / len(ranks)
        results['avg_similarity'] = sum(similarities) / len(similarities)
    
    # 计算准确率
    results['accuracy_top1'] = results['correct_top1'] / results['total_tests']
    results['accuracy_top3'] = results['correct_top3'] / results['total_tests']
    results['accuracy_top5'] = results['correct_top5'] / results['total_tests']
    
    # 打印总结
    print("\n" + "=" * 50)
    print("置信度分析总结")
    print("=" * 50)
    print(f"总测试数: {results['total_tests']}")
    print(f"Top-1 准确率: {results['accuracy_top1']:.2%}")
    print(f"Top-3 准确率: {results['accuracy_top3']:.2%}")
    print(f"Top-5 准确率: {results['accuracy_top5']:.2%}")
    print(f"平均排名: {results['avg_rank']:.2f}")
    print(f"平均相似度: {results['avg_similarity']:.4f}")
    
    return results

def run_comprehensive_analogy_analysis(analyzer):
    """
    运行全面的词类比分析
    
    Args:
        analyzer: WordSimilarityAnalyzer实例
    """
    # 定义全面的测试用例
    comprehensive_tests = [
        # 性别关系
        (['king', 'man'], 'woman', 'queen'),
        (['actor', 'man'], 'woman', 'actress'),
        (['prince', 'man'], 'woman', 'princess'),
        
        # 国家-首都
        (['paris', 'france'], 'germany', 'berlin'),
        (['london', 'england'], 'france', 'paris'),
        (['tokyo', 'japan'], 'china', 'beijing'),
        
        # 动词时态
        (['running', 'run'], 'walk', 'walking'),
        (['swam', 'swim'], 'drink', 'drank'),
        (['thought', 'think'], 'know', 'knew'),
        
        # 复数形式
        (['dogs', 'dog'], 'cat', 'cats'),
        (['children', 'child'], 'adult', 'adults'),
        
        # 比较级
        (['bigger', 'big'], 'small', 'smaller'),
        (['happier', 'happy'], 'sad', 'sadder'),
    ]
    
    print("开始全面的词类比推理分析")
    print("=" * 60)
    
    # 执行置信度分析
    confidence_results = analogy_confidence_analysis(analyzer, comprehensive_tests, topn=5)
    
    return confidence_results

def test_multiple_semantic_relations(analyzer):
    """
    测试多种语义关系
    
    Args:
        analyzer: WordSimilarityAnalyzer实例
    """
    # 1. 性别关系测试
    print("\n" + "="*60)
    print("性别关系测试")
    print("="*60)
    gender_relation = ['man', 'woman']
    gender_test_pairs = ['king', 'queen', 'prince', 'princess', 'actor', 'actress']
    analogy_strength_analysis(analyzer, gender_relation, gender_test_pairs)
    
    # 2. 国家-首都关系测试
    print("\n" + "="*60)
    print("国家-首都关系测试")
    print("="*60)
    country_capital_relation = ['france', 'paris']  # 国家→首都
    country_test_pairs = ['germany', 'berlin', 'japan', 'tokyo', 'china', 'beijing']
    analogy_strength_analysis(analyzer, country_capital_relation, country_test_pairs)
    
    # 3. 动词时态关系测试
    print("\n" + "="*60)
    print("动词时态关系测试")
    print("="*60)
    tense_relation = ['run', 'running']  # 原形→进行时
    tense_test_pairs = ['walk', 'walking', 'swim', 'swimming', 'eat', 'eating']
    analogy_strength_analysis(analyzer, tense_relation, tense_test_pairs)
    
    # 4. 单复数关系测试
    print("\n" + "="*60)
    print("单复数关系测试")
    print("="*60)
    plural_relation = ['dog', 'dogs']  # 单数→复数
    plural_test_pairs = ['cat', 'cats', 'child', 'children', 'mouse', 'mice']
    analogy_strength_analysis(analyzer, plural_relation, plural_test_pairs)

def calculate_relation_strength(analyzer, relation_pairs, relation_name="关系"):
    """
    计算语义关系的平均强度
    
    Args:
        analyzer: WordSimilarityAnalyzer实例
        relation_pairs: 关系词对列表，如 [('king','queen'), ('man','woman'), ...]
        relation_name: 关系名称
    """
    print(f"\n{relation_name}强度统计")
    print("-" * 40)
    
    success_count = 0
    total_rank = 0
    total_similarity = 0
    successful_pairs = []
    
    for word1, word2 in relation_pairs:
        # 测试该关系对
        base_relation = [word1, word2]
        test_words = [word1, word2]  # 测试自身
        
        results = analogy_strength_analysis(analyzer, base_relation, test_words)
        
        # 分析结果
        key = f"{word1}→{word2}"
        if key in results and results[key]['success']:
            success_count += 1
            total_rank += results[key]['found_rank']
            total_similarity += results[key]['found_similarity']
            successful_pairs.append((word1, word2, results[key]['found_rank'], results[key]['found_similarity']))
    
    # 输出统计结果
    if success_count > 0:
        avg_rank = total_rank / success_count
        avg_similarity = total_similarity / success_count
        success_rate = success_count / len(relation_pairs)
        
        print(f"\n📊 {relation_name}强度统计结果:")
        print(f"  测试词对数量: {len(relation_pairs)}")
        print(f"  成功识别数量: {success_count}")
        print(f"  识别成功率: {success_rate:.2%}")
        print(f"  平均排名: {avg_rank:.2f}")
        print(f"  平均相似度: {avg_similarity:.4f}")
        
        print(f"\n成功识别的词对:")
        for word1, word2, rank, sim in successful_pairs:
            print(f"  {word1} → {word2}: 排名{rank}, 相似度{sim:.4f}")
    else:
        print("没有成功识别的词对")

def main():
    """主测试程序"""

    
    print("初始化词向量分析器...")
    analyzer = WordSimilarityAnalyzer('glove-wiki-gigaword-50')
    
    # 测试单词列表
    word_list = [
        "frog", "frogs", "toad", "litoria", "leptodactylidae", "rana", 
        "lizard", "eleutherodactylus", "china", "procelin", "art", 
        "tomorrow", "today", "Tuesday", "student", "school"
    ]
    
    # 1. 执行两两相似度分析
    print("执行两两相似度分析...")
    similarity_report = pairwise_similarity_analysis(analyzer, word_list)
    
    # 2. 基于相似度进行聚类
    print("\n" + "="*60)
    print("执行聚类分析...")
    print("="*60)
    
    # 方法1: 层次聚类 (自动确定聚类数量)
    cluster_result1 = cluster_words_by_similarity(
        analyzer, 
        word_list, 
        method='hierarchical',
        n_clusters=None  # 自动确定
    )
    
    # 方法2: DBSCAN聚类 (基于密度)
    cluster_result2 = cluster_words_by_similarity(
        analyzer,
        word_list,
        method='dbscan',
        eps=0.8,  # 通过kwargs传递
        min_samples=2  # 通过kwargs传递
    )
    
    # 3. 词类比推理分析
    print("\n" + "="*60)
    print("执行词类比推理分析...")
    print("="*60)
    
    # 基本词类比推理
    print("=== 基本词类比推理 ===")
    results, success = word_analogy_analysis(
        analyzer, 
        word_pair=['king', 'man'], 
        single_word='woman',
        description="经典性别关系类比"
    )
    
    # 批量测试
    print("\n=== 批量词类比推理 ===")
    analogy_tests = [
        (['king', 'man'], 'woman', "国王-男人+女人"),
        (['paris', 'france'], 'germany', "巴黎-法国+德国"),
        (['walked', 'walk'], 'run', "走路过去式-走路+跑"),
        (['big', 'bigger'], 'small', "大-更大+小"),
    ]
    
    batch_results = batch_word_analogy(analyzer, analogy_tests)
    
    # 关系强度分析（修正版）
    print("\n=== 修正后的关系强度分析 ===")
    gender_relation = ['man', 'woman']
    gender_test_pairs = ['king', 'queen', 'boy', 'girl', 'prince', 'princess']
    strength_results = analogy_strength_analysis(analyzer, gender_relation, gender_test_pairs)
    
    print("\n所有测试完成！")



if __name__ == '__main__':

    # advanced_analyzer = AdvancedWordSimilarityAnalyzer()

    # # 计算语义关系强度
    # relation_pairs = [('king', 'queen'), ('man', 'woman'), ('boy', 'girl')]
    # strength = advanced_analyzer.semantic_relationship_strength(relation_pairs)
    # print(f"性别关系强度: {strength:.4f}")

    # # 寻找语义轴
    # gender_axis = advanced_analyzer.find_semantic_axis(
    #     positive_examples=['king', 'man', 'boy'],
    #     negative_examples=['queen', 'woman', 'girl']
    # )

    # # 投影测试
    # projection = advanced_analyzer.project_on_axis('prince', gender_axis)
    # print(f"'prince' 在性别轴上的投影: {projection:.4f}")
    main()