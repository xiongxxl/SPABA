import torch
def get_atom_error(A, B):

    if torch.all(B[A==1]==1):
        atom_error=int(sum(B).sum()-A.sum().item())

    else:
        atom_error=-1

    return atom_error


if __name__ == "__main__":
    A = torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0')
    B = torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0')
    atom_error=get_atom_error(A,B)
    print(atom_error)