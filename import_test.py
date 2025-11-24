import sys
import traceback
from pathlib import Path


# Try to find the ai_face_persona package/directory in a robust, cross-platform way:
# 1) project_root/ai_face_persona
# 2) search upward a few parent directories for ai_face_persona
# 3) common fallbacks (Windows and macOS/Linux Downloads/Persona)
def find_ai_face_persona(start: Path, max_up: int = 4):
    candidates = []

    # direct sibling folder
    candidates.append(start / 'ai_face_persona')

    # search upwards for folder named ai_face_persona
    p = start
    for _ in range(max_up):
        candidate = p / 'ai_face_persona'
        candidates.append(candidate)
        p = p.parent

    # historical Windows fallback from original project
    candidates.append(Path(r'c:\Users\Tuba Khan\Downloads\Persona\ai_face_persona'))

    # common Downloads fallback for macOS/Linux
    candidates.append(Path.home() / 'Downloads' / 'Persona' / 'ai_face_persona')

    for c in candidates:
        try:
            if c.exists():
                return c
        except Exception:
            # Some Path.exists() may fail for weird paths; ignore and continue
            continue

    return None


project_root = Path(__file__).resolve().parent
ai_face_persona = find_ai_face_persona(project_root)

if ai_face_persona:
    # Insert the ai_face_persona folder itself so `import face_detector` works if modules are top-level
    sys.path.insert(0, str(ai_face_persona))
    print(f'Added to sys.path: {ai_face_persona}')
else:
    print(f'Warning: ai_face_persona not found under {project_root} or common fallback paths')


try:
    import face_detector
    print('face_detector imported')
    import overlay_utils
    print('overlay_utils imported')
    import emotion_model
    print('emotion_model imported')
    print('Imports OK')
except Exception:
    traceback.print_exc()
    print('Import test failed')
