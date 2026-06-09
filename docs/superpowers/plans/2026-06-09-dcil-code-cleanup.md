# DCIL Code Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the reused SGIL/HIL code into a clean PyTorch repository whose public API and filenames match the paper method: Dual-Consistency Invariant Learning (DCIL).

**Architecture:** Keep datasets and evaluation utilities, remove old TensorFlow and experimental model branches, and expose a single PyTorch DCIL implementation backed by LightGCN-S. DCIL is decomposed into BaseCF, LightGCNS, SocialAwareViewGenerator, and DCIL so the paper components are visible in code.

**Tech Stack:** Python, PyTorch, NumPy, SciPy, FAISS optional for evaluation acceleration.

---

### Task 1: Repository Contract Tests

**Files:**
- Create: `tests/test_repository_contract.py`

- [x] Write tests that fail until the repository exposes DCIL names, removes old SGIL/HIL files, and documents the paper workflow.
- [ ] Run `python tests/test_repository_contract.py` or an equivalent unittest/pytest command and confirm red failure.
- [ ] Implement repository cleanup and DCIL files.
- [ ] Run the contract tests and syntax checks until green.
- [ ] Commit the verified stage.

### Task 2: DCIL Implementation

**Files:**
- Create/modify: `models/base_model.py`, `models/lightgcn_s.py`, `models/view_generator.py`, `models/dcil.py`, `models/__init__.py`
- Modify: `run_DCIL.py`, `utils/rec_dataset.py`, `README.md`, `.gitignore`, `requirements.txt`

- [ ] Implement paper-aligned names: `BaseCF`, `LightGCNS`, `SocialAwareViewGenerator`, `DCIL`.
- [ ] Implement semantic social intensity coefficients, Gumbel-Softmax masks, RC loss, PC loss, and InfoNCE recommendation loss.
- [ ] Keep training/evaluation lightweight and compatible with `douban_book`, `yelp`, and `epinions`.
- [ ] Remove legacy TensorFlow SGIL, HIL experiments, pycache, and notebook checkpoint files from the main repo.
- [ ] Commit the verified stage.

### Task 3: GitHub Handoff

**Files:**
- Git metadata only.

- [ ] Initialize git if missing.
- [ ] Add remote `https://github.com/bang51666/Dual-Consistency-Invariant-Learning-for-Social-Recommendation-DCIL-.git`.
- [ ] Push staged commits to GitHub.
