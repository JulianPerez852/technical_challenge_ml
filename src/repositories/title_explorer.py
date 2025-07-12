import pandas as pd
import numpy as np
import re
import json
import os
import sys
import logging
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from exceptions.exceptions import DataLoadingException, DataAnalysisException, DataSavingException
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from exceptions.exceptions import DataLoadingException, DataAnalysisException, DataSavingException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TitleAnalyzer:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.words_new = Counter()
        self.words_used = Counter()
        self.identified_patterns = {}
        self.identified_categories = {}
        self.identified_brands = {}
        self.key_words_new = []
        self.key_words_used = []
        self.analysis_results = {}
        
        logger.info(f"TitleAnalyzer initialized with input: {input_path}, output: {output_path}")
        
    def load_data(self):
        """Load dataset from CSV file"""
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info(f"Dataset loaded successfully: {len(self.df)} records")
            
            if 'title' not in self.df.columns or 'condition' not in self.df.columns:
                raise DataLoadingException("Dataset must contain 'title' and 'condition' columns")
            
            self.df['title'] = self.df['title'].fillna('').astype(str)
            self.df['condition'] = self.df['condition'].fillna('').astype(str)
            
            new_count = len(self.df[self.df['condition'] == 'new'])
            used_count = len(self.df[self.df['condition'] == 'used'])
            logger.info(f"Dataset distribution - NEW: {new_count}, USED: {used_count}")
            
        except Exception as e:
            logger.error(f"Error loading dataset from {self.input_path}")
            raise DataLoadingException(f"Error loading dataset from {self.input_path}") from e
    
    def basic_text_cleaning(self, text):
        """Basic text cleaning"""
        if pd.isna(text):
            return ""
        return str(text).strip()
    
    def calculate_basic_statistics(self):
        """Calculate basic statistics for titles"""
        try:
            logger.info("Calculating basic statistics for titles")
            
            stats = {}
            for condition in ['new', 'used']:
                titles_cond = self.df[self.df['condition'] == condition]['title']
                
                lengths = titles_cond.str.len()
                words = titles_cond.str.split().str.len()
                
                stats[condition] = {
                    'count': len(titles_cond),
                    'avg_length': lengths.mean(),
                    'median_length': lengths.median(),
                    'avg_words': words.mean(),
                    'median_words': words.median(),
                    'max_length': lengths.max(),
                    'min_length': lengths.min()
                }
                
                logger.info(f"{condition.upper()} statistics - Count: {stats[condition]['count']}, "
                        f"Avg length: {stats[condition]['avg_length']:.1f}, "
                        f"Avg words: {stats[condition]['avg_words']:.1f}")
            
            self.analysis_results['basic_stats'] = stats
            
        except Exception as e:
            logger.error("Error calculating basic statistics")
            raise DataAnalysisException("Error calculating basic statistics") from e
    
    def extract_words_by_condition(self):
        """Extract most frequent words by condition"""
        try:
            logger.info("Extracting words by condition")
            
            titles_new = self.df[self.df['condition'] == 'new']['title']
            titles_used = self.df[self.df['condition'] == 'used']['title']
            
            def extract_words(titles):
                all_words = []
                for title in titles:
                    clean_title = re.sub(r'[^\w\s]', ' ', str(title).lower())
                    words = clean_title.split()
                    words = [w for w in words if len(w) > 2]
                    all_words.extend(words)
                return Counter(all_words)
            
            self.words_new = extract_words(titles_new)
            self.words_used = extract_words(titles_used)
            
            logger.info(f"Words extracted - NEW: {len(self.words_new)} unique words, "
                    f"USED: {len(self.words_used)} unique words")
            

            self.analysis_results['top_words_new'] = self.words_new.most_common(20)
            self.analysis_results['top_words_used'] = self.words_used.most_common(20)
            
            self.identify_distinctive_words()
            
        except Exception as e:
            logger.error("Error extracting words by condition")
            raise DataAnalysisException("Error extracting words by condition") from e
    
    def identify_distinctive_words(self):
        """Identify words that are more characteristic of each condition"""
        try:
            logger.info("Identifying distinctive words")
            
            distinctive_new = []
            distinctive_used = []
            

            all_words = set(list(self.words_new.keys()) + list(self.words_used.keys()))
            
            for word in all_words:
                freq_new = self.words_new.get(word, 0)
                freq_used = self.words_used.get(word, 0)
                
                if freq_new + freq_used >= 2:

                    ratio_new = freq_new / (freq_used + 1)
                    ratio_used = freq_used / (freq_new + 1)
                    
                    if ratio_new > 2 and freq_new >= 2: 
                        distinctive_new.append((word, freq_new, freq_used, ratio_new))
                    elif ratio_used > 2 and freq_used >= 2: 
                        distinctive_used.append((word, freq_used, freq_new, ratio_used))
            
            distinctive_new.sort(key=lambda x: x[3], reverse=True)
            distinctive_used.sort(key=lambda x: x[3], reverse=True)
            
            self.key_words_new = [item[0] for item in distinctive_new[:20]]
            self.key_words_used = [item[0] for item in distinctive_used[:20]]
            
            logger.info(f"Distinctive words identified - NEW: {len(self.key_words_new)}, "
                    f"USED: {len(self.key_words_used)}")

            self.analysis_results['distinctive_words_new'] = distinctive_new[:15]
            self.analysis_results['distinctive_words_used'] = distinctive_used[:15]
            
        except Exception as e:
            logger.error("Error identifying distinctive words")
            raise DataAnalysisException("Error identifying distinctive words") from e
    
    def detect_numeric_patterns(self):
        """Detect numeric patterns and years"""
        try:
            logger.info("Detecting numeric patterns")

            patterns = {
                'years': r'\b(19|20)\d{2}\b',
                'specifications_gb': r'\b\d+\s*gb\b',
                'specifications_mb': r'\b\d+\s*mb\b',
                'specifications_mpx': r'\b\d+\s*mpx\b',
                'specifications_core': r'\b(core\s*i\d|i\d)\b',
                'specifications_ram': r'\b\d+\s*ram\b',
                'numeric_models': r'\b\d{3,4}\b',
                'versions': r'\bv\d+\b',
                'dimensions': r'\b\d+\s*(inches|"|\'\')?\b'
            }
            
            pattern_results = {}
            
            for pattern_name, pattern_regex in patterns.items():
                pattern_results[pattern_name] = {'new': [], 'used': []}
                
                for condition in ['new', 'used']:
                    titles = self.df[self.df['condition'] == condition]['title']
                    matches = []
                    
                    for title in titles:
                        found = re.findall(pattern_regex, str(title).lower())
                        matches.extend(found)
                    
                    pattern_results[pattern_name][condition] = matches
                
                logger.info(f"Pattern '{pattern_name}' - NEW: {len(pattern_results[pattern_name]['new'])} matches, "
                        f"USED: {len(pattern_results[pattern_name]['used'])} matches")
            
            self.analysis_results['numeric_patterns'] = pattern_results
            self.analyze_years_detailed()
            
        except Exception as e:
            logger.error("Error detecting numeric patterns")
            raise DataAnalysisException("Error detecting numeric patterns") from e
    
    def analyze_years_detailed(self):
        """Detailed analysis of years found"""
        try:
            logger.info("Analyzing years in detail")
            
            years_by_condition = {'new': [], 'used': []}
            
            for condition in ['new', 'used']:
                titles = self.df[self.df['condition'] == condition]['title']
                
                for title in titles:
                    years = re.findall(r'\b(19|20)\d{2}\b', str(title))
                    years_by_condition[condition].extend([int(year) for year in years])
            
            year_stats = {}
            for condition in ['new', 'used']:
                years = years_by_condition[condition]
                if years:
                    year_stats[condition] = {
                        'mean': np.mean(years),
                        'median': np.median(years),
                        'min': min(years),
                        'max': max(years),
                        'most_common': Counter(years).most_common(5)
                    }
                    
                    logger.info(f"Years analysis for {condition.upper()} - "
                            f"Mean: {year_stats[condition]['mean']:.0f}, "
                            f"Range: {year_stats[condition]['min']}-{year_stats[condition]['max']}")
            
            self.analysis_results['years_analysis'] = year_stats
            
        except Exception as e:
            logger.error("Error analyzing years")
            raise DataAnalysisException("Error analyzing years") from e
    
    def identify_brands_automatically(self):
        """Identify possible brands automatically"""
        try:
            logger.info("Identifying brands automatically")
            
            possible_brands = Counter()
            
            for title in self.df['title']:
                title_str = str(title)
                
                uppercase_words = re.findall(r'\b[A-Z][A-Z]+\b', title_str)
                possible_brands.update([w.lower() for w in uppercase_words])

                capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', title_str)
                possible_brands.update([w.lower() for w in capitalized_words])
            

            brand_candidates = [(brand, freq) for brand, freq in possible_brands.most_common(50) if freq >= 3]
            
            logger.info(f"Identified {len(brand_candidates)} brand candidates")

            brand_distribution = {}
            for brand, _ in brand_candidates[:20]:
                freq_new = sum(1 for title in self.df[self.df['condition'] == 'new']['title'] 
                            if brand in str(title).lower())
                freq_used = sum(1 for title in self.df[self.df['condition'] == 'used']['title'] 
                            if brand in str(title).lower())
                
                total = freq_new + freq_used
                if total > 0:
                    brand_distribution[brand] = {
                        'new': freq_new,
                        'used': freq_used,
                        'total': total,
                        'ratio_new': freq_new / total
                    }
            
            self.identified_brands = dict(brand_candidates[:30])
            self.analysis_results['brand_candidates'] = brand_candidates[:20]
            self.analysis_results['brand_distribution'] = brand_distribution
            
        except Exception as e:
            logger.error("Error identifying brands")
            raise DataAnalysisException("Error identifying brands") from e
    
    def identify_categories_automatically(self):
        """Identify product categories automatically"""
        try:
            logger.info("Identifying product categories automatically")

            category_indicators = {
                'electronics': ['celular', 'smartphone', 'notebook', 'laptop', 'tablet', 'tv', 'televisor',
                            'monitor', 'auriculares', 'parlante', 'camara', 'iphone', 'samsung', 'display'],
                'automotive': ['auto', 'carro', 'vehiculo', 'motor', 'llanta', 'amortiguador', 'filtro',
                            'aceite', 'freno', 'bateria', 'repuesto', 'ford', 'chevrolet', 'toyota'],
                'books': ['libro', 'enciclopedia', 'manual', 'guia', 'diccionario', 'atlas', 'historia',
                        'novela', 'biografia', 'ensayo', 'literatura', 'editorial'],
                'clothing': ['zapatillas', 'zapatos', 'remera', 'pantalon', 'jean', 'camisa', 'vestido',
                            'buzo', 'campera', 'nike', 'adidas', 'talle', 'ropa'],
                'home': ['mesa', 'silla', 'cama', 'sofa', 'armario', 'espejo', 'lampara', 'decoracion',
                        'cocina', 'baño', 'jardin', 'mueble', 'fundas', 'almohada'],
                'toys': ['juguete', 'muneca', 'pelota', 'puzzle', 'rompecabezas', 'lego', 'barbie',
                        'disney', 'juego', 'infantil', 'niño', 'niña'],
                'sports': ['bicicleta', 'pelota', 'futbol', 'tenis', 'gym', 'fitness', 'deportivo',
                        'ejercicio', 'running', 'natacion'],
                'music': ['guitarra', 'piano', 'violin', 'bateria', 'microfono', 'amplificador',
                        'instrumento', 'musical', 'fender', 'yamaha']
            }
            

            category_stats = {}
            
            for category, keywords in category_indicators.items():
                count_new = 0
                count_used = 0
                
                for condition in ['new', 'used']:
                    titles = self.df[self.df['condition'] == condition]['title']
                    
                    for title in titles:
                        title_lower = str(title).lower()
                        if any(keyword in title_lower for keyword in keywords):
                            if condition == 'new':
                                count_new += 1
                            else:
                                count_used += 1
                
                total = count_new + count_used
                if total > 0:
                    category_stats[category] = {
                        'new': count_new,
                        'used': count_used,
                        'total': total,
                        'ratio_new': count_new / total
                    }
            
            logger.info(f"Identified {len(category_stats)} product categories")
            
            self.identified_categories = category_stats
            self.analysis_results['category_stats'] = category_stats
            
        except Exception as e:
            logger.error("Error identifying categories")
            raise DataAnalysisException("Error identifying categories") from e
    
    def detect_special_patterns(self):
        """Detect special patterns in titles"""
        try:
            logger.info("Detecting special patterns")
            
            special_patterns = {
                'exclamation_marks': r'!+',
                'uppercase_words': r'\b[A-Z]{3,}\b',
                'numbers_parentheses': r'\([^)]*\d[^)]*\)',
                'multiple_hyphens': r'-{2,}',
                'peso_prices': r'\$\d+',
                'measurements': r'\b\d+\s*(cm|mm|m|inches|")\b',
                'colors': r'\b(negro|blanco|azul|rojo|verde|amarillo|gris|rosa|violeta)\b',
                'conditions': r'\b(nuevo|usado|seminuevo|impecable|excelente|bueno|regular)\b',
                'urgency': r'\b(urgente|liquido|oferta|ganga|oportunidad)\b'
            }
            
            pattern_stats = {}
            
            for pattern_name, pattern_regex in special_patterns.items():
                pattern_stats[pattern_name] = {}
                
                for condition in ['new', 'used']:
                    titles = self.df[self.df['condition'] == condition]['title']
                    matches = 0
                    
                    for title in titles:
                        if re.search(pattern_regex, str(title), re.IGNORECASE):
                            matches += 1
                    
                    pattern_stats[pattern_name][condition] = matches
                
                logger.info(f"Special pattern '{pattern_name}' - "
                        f"NEW: {pattern_stats[pattern_name]['new']}, "
                        f"USED: {pattern_stats[pattern_name]['used']}")
            
            self.analysis_results['special_patterns'] = pattern_stats
            
        except Exception as e:
            logger.error("Error detecting special patterns")
            raise DataAnalysisException("Error detecting special patterns") from e
    
    def generate_complete_report(self):
        """Generate a complete report with all identified variables"""
        try:
            logger.info("Generating complete analysis report")

            complete_report = {
                'dataset_info': {
                    'total_records': len(self.df),
                    'new_records': len(self.df[self.df['condition'] == 'new']),
                    'used_records': len(self.df[self.df['condition'] == 'used'])
                },
                'key_words_new': self.key_words_new,
                'key_words_used': self.key_words_used,
                'main_brands': list(self.identified_brands.keys()),
                'product_categories': list(self.identified_categories.keys()),
                'regex_patterns': {
                    'years': r'\b(19|20)\d{2}\b',
                    'specifications_gb': r'\b\d+\s*gb\b',
                    'specifications_core': r'\b(core\s*i\d|i\d)\b',
                    'numeric_models': r'\b\d{3,4}\b',
                    'uppercase_words': r'\b[A-Z]{3,}\b',
                    'urgency': r'\b(urgente|liquido|oferta|ganga)\b'
                },
                'analysis_results': self.analysis_results
            }
            
            logger.info("Complete analysis report generated successfully")
            return complete_report
            
        except Exception as e:
            logger.error("Error generating complete report")
            raise DataAnalysisException("Error generating complete report") from e
    
    def run_complete_analysis(self):
        """Execute complete automatic analysis"""
        try:
            logger.info("Starting complete title analysis")
            
            if self.df is None:
                logger.error("No data loaded. Please load data first.")
                raise DataAnalysisException("No data loaded. Please load data first.")
            
            self.calculate_basic_statistics()
            self.extract_words_by_condition()
            self.detect_numeric_patterns()
            self.identify_brands_automatically()
            self.identify_categories_automatically()
            self.detect_special_patterns()
            
            complete_report = self.generate_complete_report()
            
            logger.info("Complete analysis finished successfully")
            return complete_report
            
        except Exception as e:
            logger.error("Error during complete analysis")
            raise DataAnalysisException("Error during complete analysis") from e
    
    def save_results(self, results):
        """Save analysis results to JSON file"""
        try:
            logger.info(f"Saving analysis results to {self.output_path}")

            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Analysis results saved successfully to {self.output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results to {self.output_path}: {str(e)}")
            raise DataSavingException(f"Error saving results to {self.output_path}: {str(e)}") from e

if __name__ == "__main__":
    try:
        input_path = '../../data/x_train.csv'
        output_path = '../../data/title_analysis_results.json'
        
        analyzer = TitleAnalyzer(input_path, output_path)
        analyzer.load_data()
        results = analyzer.run_complete_analysis()
        analyzer.save_results(results)
        
    except (DataLoadingException, DataAnalysisException, DataSavingException) as e:
        logger.error(f"Error in title analysis: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)