import fitz
import os

pdf_path = "/home/adrian/projects/ai-inventor/aii_data/users/admin/runs/run_v4q390i9efU9/4_gen_paper_repo/_4_assemble_paper/paper/workspace/paper.pdf"
out_dir = "/home/adrian/projects/ai-inventor/aii_data/users/admin/runs/run_v4q390i9efU9/4_gen_paper_repo/_4_assemble_paper/paper/workspace/pages"
os.makedirs(out_dir, exist_ok=True)

doc = fitz.open(pdf_path)
for i, page in enumerate(doc):
    mat = fitz.Matrix(150/72, 150/72)
    pix = page.get_pixmap(matrix=mat)
    out_path = os.path.join(out_dir, f"page_{i+1:02d}.png")
    pix.save(out_path)
    print(f"Saved {out_path}")
print(f"Total pages: {len(doc)}")
