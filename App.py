import sys, traceback, os
# Add ai_face_persona folder (next to this script) to sys.path so 'main' can be imported
repo_root = os.path.dirname(__file__)
ai_path = os.path.join(repo_root, 'persona')
if os.path.isdir(ai_path):
    sys.path.insert(0, ai_path)
else:
    # fallback: add repo root so a top-level main.py can be found
    sys.path.insert(0, repo_root)

try:
    # Try a normal import first (works when package is installed or in PYTHONPATH)
    try:
        import main
    except Exception:
        # Fallback: load main.py directly from ai_face_persona/ or repo root
        import importlib.util

        def _load_main_from_paths():
            candidates = [
                os.path.join(repo_root, 'persona', 'main.py'),
                os.path.join(repo_root, 'main.py'),
            ]
            for path in candidates:
                if os.path.isfile(path):
                    spec = importlib.util.spec_from_file_location('main', path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module
            raise ImportError("Could not find a 'main.py' in persona/ or repo root")

        main = _load_main_from_paths()

    print('Starting main()')
    main.main()
except Exception:
    traceback.print_exc()
    print('ERROR launching main')
