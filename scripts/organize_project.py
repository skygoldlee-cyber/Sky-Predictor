#!/usr/bin/env python
"""
프로젝트 폴더 정리 스크립트

docs, prediction, tests 폴더를 자동으로 정리합니다.
"""

import os
import shutil
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent

def organize_docs():
    """docs 폴더 정리"""
    docs_path = PROJECT_ROOT / "docs"
    
    # 하위 폴더 생성
    subdirs = ["guides", "architecture", "reviews", "reports", "archives"]
    for subdir in subdirs:
        (docs_path / subdir).mkdir(exist_ok=True)
    
    # 파일 분류
    guides_files = [
        "ADAPTIVE_INDICATOR_GUIDE.md",
        "CONFIG_REFERENCE_GUIDE.md",
        "PARAMETER_TUNING_GUIDE.md",
        "DUAL_MODE_GUIDE.md",
        "RUNTIME_VS_BACKTEST_GUIDE.md",
        "POSITION_SIZING_GUIDE.md",
        "TRADE_LOGGING_GUIDE.md",
        "TRADING_SIGNAL_GENERATION_GUIDE.md",
        "CONFORMAL_PREDICTION_GUIDE.md",
        "GUARDRAIL_SYSTEM_GUIDE.md",
        "INCIDENT_HANDLING_GUIDE.md",
        "FEEDBACK_SYSTEM_GUIDE.md",
        "LLM_JUDGE_SYSTEM_GUIDE.md",
        "MULTISCALE_FEATURES_GUIDE.md",
        "MULTITIMEFRAME_FEATURES.md",
        "OPTION_FLOW_ANALYSIS_GUIDE.md",
        "OPTION_SENTIMENT_INTEGRATION_GUIDE.md",
        "PIVOT_COLLECTOR_GUIDE.md",
        "PIVOT_ML_ALGORITHM_GUIDE.md",
        "PERFORMANCE_ANALYSIS_GUIDE.md",
        "MODELS_GUIDE.md",
        "RUNTIME_API_REFERENCE.md",
    ]
    
    architecture_files = [
        "Architecture.md",
        "SYSTEM_ALGORITHM_OVERVIEW.md",
        "SYSTEM_ALGORITHM_OVERVIEW.html",
        "MODEL_TRAINING_GUIDE.md",
        "MODEL_TRAINING_GUIDE.html",
        "dataset_training_guide.md",
        "dataset_training_guide.html",
        "market_regime_classifier.md",
        "market_regime_classifier.html",
        "zigzag_pivot_logic.md",
        "zigzag_pivot_logic.html",
    ]
    
    reviews_files = [
        "code_review_report.md",
        "prediction_code_review.md",
        "prediction_algorithm_review.md",
        "source_review_v2.md",
        "adaptive_indicator_analysis.md",
        "adaptive_indicator_deep_review.md",
        "Transformer_코드리뷰.md",
        "telegram_review.md",
        "ZigZag_Pivot_Analysis.md",
    ]
    
    reports_files = [
        "BUG_FIXES.md",
        "optimize_zigzag_lag_improvements.md",
        "zigzag_pivot_improvement.md",
        "PIVOT_SIGNAL_IMPROVEMENTS.md",
        "improvement_report.md",
        "transformer_improvement_report.md",
        "Transformer_보완_개선_리포트.md",
        "SkyPredictor_개선사항.md",
        "adaptive_zigzag_fixes.md",
        "ZigZag_Pivot_Principles_BugFix.md",
    ]
    
    archives_files = [
        "pivot_confirm_debug.md",
        "pivot_confirmation_logic.md",
        "pivot_confirmation_logic_merged.md",
        "pivot_info_panel_chart_internal_placement.md",
        "pivot_info_panel_implementation.md",
        "daily_train_patch_tst.md",
        "t8415_minute_data_logic.md",
        "hardcoded_values.md",
        "zigzag_param_unification_report.md",
        "zigzag_pivot_configuration_guide.md",
    ]
    
    # 파일 이동
    for file in guides_files:
        src = docs_path / file
        if src.exists():
            shutil.move(src, docs_path / "guides" / file)
    
    for file in architecture_files:
        src = docs_path / file
        if src.exists():
            shutil.move(src, docs_path / "architecture" / file)
    
    for file in reviews_files:
        src = docs_path / file
        if src.exists():
            shutil.move(src, docs_path / "reviews" / file)
    
    for file in reports_files:
        src = docs_path / file
        if src.exists():
            shutil.move(src, docs_path / "reports" / file)
    
    for file in archives_files:
        src = docs_path / file
        if src.exists():
            shutil.move(src, docs_path / "archives" / file)
    
    print("✅ docs 폴더 정리 완료")

def organize_prediction():
    """prediction 폴더 정리"""
    prediction_path = PROJECT_ROOT / "prediction"
    
    # 하위 폴더 생성
    subdirs = ["models", "features", "mixins", "training", "backtest"]
    for subdir in subdirs:
        (prediction_path / subdir).mkdir(exist_ok=True)
    
    # 파일 분류
    models_files = [
        "model.py",
        "tft_model.py",
        "mamba_model.py",
        "pivot_models.py",
    ]
    
    features_files = [
        "features.py",
        "time_features.py",
        "oi_features.py",
        "option_features.py",
        "option_flow_features.py",
        "parity_features.py",
        "similarity_features.py",
    ]
    
    mixins_files = [
        "adaptive_mixin.py",
        "prediction_mixin.py",
        "llm_mixin.py",
        "amplitude_mixin.py",
        "feedback_mixin.py",
        "guardrail_mixin.py",
        "option_mixin.py",
        "tick_mixin.py",
    ]
    
    training_files = [
        "train_pivot_classifier.py",
        "train_pivot_regressor.py",
        "train_pivot_lifespan.py",
    ]
    
    backtest_files = [
        "backtest_pivot_signals.py",
        "zigzag_backtester.py",
    ]
    
    # 파일 이동
    for file in models_files:
        src = prediction_path / file
        if src.exists():
            shutil.move(src, prediction_path / "models" / file)
    
    for file in features_files:
        src = prediction_path / file
        if src.exists():
            shutil.move(src, prediction_path / "features" / file)
    
    for file in mixins_files:
        src = prediction_path / file
        if src.exists():
            shutil.move(src, prediction_path / "mixins" / file)
    
    for file in training_files:
        src = prediction_path / file
        if src.exists():
            shutil.move(src, prediction_path / "training" / file)
    
    for file in backtest_files:
        src = prediction_path / file
        if src.exists():
            shutil.move(src, prediction_path / "backtest" / file)
    
    print("✅ prediction 폴더 정리 완료")

def organize_tests():
    """tests 폴더 정리"""
    tests_path = PROJECT_ROOT / "tests"
    
    # 하위 폴더 생성
    subdirs = ["gui", "prediction"]
    for subdir in subdirs:
        (tests_path / subdir).mkdir(exist_ok=True)
    
    # 파일 분류
    gui_files = [
        "test_chart_viewer.py",
        "test_chart_viewer_time_index.py",
        "test_gui_controller_config_reload.py",
        "test_gui_controller_market.py",
        "test_gui_controller_rt_helpers.py",
    ]
    
    prediction_files = [
        "test_prediction_smoke.py",
        "test_predictor_confidence_adjust.py",
        "test_data_builder_gz_smoke.py",
        "test_replay_gz_smoke.py",
        "test_replay_verification.py",
    ]
    
    # 파일 이동
    for file in gui_files:
        src = tests_path / file
        if src.exists():
            shutil.move(src, tests_path / "gui" / file)
    
    for file in prediction_files:
        src = tests_path / file
        if src.exists():
            shutil.move(src, tests_path / "prediction" / file)
    
    print("✅ tests 폴더 정리 완료")

def main():
    """메인 함수"""
    print("🗂️  프로젝트 폴더 정리 시작...")
    
    try:
        organize_docs()
        organize_prediction()
        organize_tests()
        print("\n✅ 모든 폴더 정리 완료!")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()
