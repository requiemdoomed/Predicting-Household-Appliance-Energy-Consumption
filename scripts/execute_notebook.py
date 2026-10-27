"""Execute the notebook, using --in-process only when kernel sockets are blocked."""
from pathlib import Path
import argparse
import os
import nbformat
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-process',action='store_true')
    args=parser.parse_args()
    os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'
    path=ROOT/'WattWise.ipynb'; nb=nbformat.read(path,as_version=4)
    if args.in_process:
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.utils.capture import capture_output
        os.chdir(ROOT)
        shell=InteractiveShell.instance(); count=0
        for cell in nb.cells:
            if cell.cell_type!='code': continue
            count+=1
            with capture_output() as captured:
                result=shell.run_cell(cell.source,store_history=True)
            if result.error_before_exec or result.error_in_exec:
                raise RuntimeError(f'Cell {count} failed: {captured.stdout} {captured.stderr}')
            outputs=[]
            if captured.stdout: outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
            if captured.stderr: outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
            for rich in captured.outputs:
                outputs.append(nbformat.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
            cell.outputs=outputs;cell.execution_count=count
        mode='fresh-process IPython (no kernel sockets)'
    else:
        from nbclient import NotebookClient
        NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
        mode='Jupyter kernel'
    nb.metadata['execution_verification']={'mode':mode}
    nbformat.validate(nb);nbformat.write(nb,path)
    print(f'Executed {sum(c.cell_type=="code" for c in nb.cells)} code cells without errors using {mode}.')

if __name__=='__main__':main()
