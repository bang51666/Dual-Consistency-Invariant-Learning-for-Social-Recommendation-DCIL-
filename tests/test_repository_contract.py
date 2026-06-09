from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_dcil_entrypoint_and_model_files_exist():
    expected = [
        "run_DCIL.py",
        "models/dcil.py",
        "models/view_generator.py",
        "models/lightgcn_s.py",
        "models/base_model.py",
        "utils/rec_dataset.py",
        "utils/evaluate.py",
        "README.md",
        ".gitignore",
    ]
    missing = [path for path in expected if not (ROOT / path).exists()]
    assert missing == []


def test_legacy_sgil_and_experimental_files_are_removed_from_main_repo():
    legacy_paths = [
        "run_SGIL.py",
        "models/SGIL.py",
        "models/PairWise_model.py",
        "torch_version/models/SGIL.py",
        "torch_version/models/HIL_Individual.py",
        "torch_version/models/HIL_Individual_VP.py",
    ]
    remaining = [path for path in legacy_paths if (ROOT / path).exists()]
    assert remaining == []


def test_model_package_exports_paper_method_names():
    init_text = read("models/__init__.py")
    for name in ["DCIL", "SocialAwareViewGenerator", "LightGCNS", "BaseCF"]:
        assert name in init_text
    for old_name in ["SGIL", "HIL_Individual", "HIL_Individual_VP", "Base_CF"]:
        assert old_name not in init_text


def test_dcil_model_defines_rc_pc_and_generator_components():
    dcil_tree = ast.parse(read("models/dcil.py"))
    class_names = {node.name for node in ast.walk(dcil_tree) if isinstance(node, ast.ClassDef)}
    method_names = {node.name for node in ast.walk(dcil_tree) if isinstance(node, ast.FunctionDef)}
    assert "DCIL" in class_names
    assert "compute_representation_consistency_loss" in method_names
    assert "compute_prediction_consistency_loss" in method_names
    assert "compute_recommendation_loss" in method_names
    assert "get_loss" in method_names


def test_run_dcil_parser_uses_paper_hyperparameter_names():
    source = read("run_DCIL.py")
    for cli_name in ["--num_views", "--lambda_rc", "--lambda_pc", "--gumbel_temperature", "--rc_temperature"]:
        assert cli_name in source
    assert "DCIL" in source
    assert "SGIL" not in source


def test_readme_describes_paper_experiment_workflow():
    readme = read("README.md")
    for phrase in [
        "Dual-Consistency Invariant Learning",
        "Social-aware View Generator",
        "Representation Consistency",
        "Prediction Consistency",
        "Douban-Book",
        "Yelp",
        "Epinions",
    ]:
        assert phrase in readme
