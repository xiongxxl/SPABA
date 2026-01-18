import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from atom_criterion import get_atom_error
from mark import atom_flags_padded
from sklearn.metrics import matthews_corrcoef
import numpy

# # train,test,valid
# def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, device, max_atom):
#     model.to(device)
#     train_losses = []
#     val_losses = []
#     train_accs = []
#     val_accs = []
#
#     best_val_acc = 0.0
#     best_model_state = None
#
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#
#         for inputs, labels, smi_length, smiles in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
#             batch_size = inputs.size(0)
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#
#             # Construct attention mask for output
#             mask_outputs = []
#             for i in range(batch_size):
#                 smiles_single = smiles[i]
#                 smiles_mask = atom_flags_padded(smiles_single)
#                 mask_outputs.append(smiles_mask)
#             mask_outputs = torch.tensor(mask_outputs, device=outputs.device)
#
#             # Apply mask to the output
#             outputs_refine = outputs * mask_outputs
#
#             # Compute loss with masking
#             loss = criterion(outputs_refine, labels)
#             mask = torch.zeros_like(loss)
#             for i in range(batch_size):
#                 k = smi_length[i]
#                 mask[i, :k] = 1
#             loss = loss * mask
#             loss = loss.sum() / smi_length.sum()
#
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#
#             # Compute training accuracy
#             outputs = torch.sigmoid(outputs)
#             predicted = (outputs >= 0.5).float()
#
#             for i in range(labels.size(0)):
#                 label_atoms = labels[i][:smi_length[i]]
#                 predicted_atoms = predicted[i][:smi_length[i]]
#                 atom_error = get_atom_error(label_atoms, predicted_atoms)
#                 if 0 <= atom_error <= max_atom:
#                     correct += 1
#                 total += 1
#
#         train_loss = running_loss / len(train_loader)
#         train_acc = 100 * correct / total
#         train_losses.append(train_loss)
#         train_accs.append(train_acc)
#
#         # Validation
#         val_loss, val_acc = evaluate_model(model, val_loader, criterion, device, max_atom)
#         val_losses.append(val_loss)
#         val_accs.append(val_acc)
#
#         print(f'Epoch {epoch + 1}/{num_epochs} - '
#               f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
#               f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
#
#         # Save the best model based on validation accuracy
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_model_state = model.state_dict()
#             print(f'✅ New best model found at Epoch {epoch + 1} with Val Acc: {val_acc:.2f}%')
#
#     # Final evaluation on the test set
#     test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, max_atom)
#     print(f'\n Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
#
#
#
#     # Load and save best model
#     if best_model_state is not None:
#         model.load_state_dict(best_model_state)
#         torch.save(model.state_dict(), 'best_model.pth')
#         print(' Best model saved as best_model.pt')
#         test_loss_best, test_acc_best = evaluate_model(model, test_loader, criterion, device, max_atom)
#         print(f'\n Test Loss best: {test_loss_best:.4f}, Test Acc: {test_acc_best:.2f}%')
#
#     return best_model_state,best_val_acc, train_losses, val_losses, train_accs, val_accs, test_loss, test_acc
#
# def evaluate_model(model, loader, criterion, device, max_atom):
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         for inputs, labels, smi_length, smiles in loader:
#             batch_size = inputs.size(0)
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#
#             loss = criterion(outputs, labels)
#             mask = torch.zeros_like(loss)
#             for i in range(batch_size):
#                 k = smi_length[i]
#                 mask[i, :k] = 1
#             loss = loss * mask
#             loss = loss.sum() / smi_length.sum()
#             running_loss += loss.item()
#
#             outputs = torch.sigmoid(outputs)
#             predicted = (outputs >= 0.5).float()
#
#             for i in range(labels.size(0)):
#                 label_atoms = labels[i][:smi_length[i]]
#                 predicted_atoms = predicted[i][:smi_length[i]]
#                 atom_error = get_atom_error(label_atoms, predicted_atoms)
#                 if 0 <= atom_error <= max_atom:
#                     correct += 1
#                 total += 1
#
#     avg_loss = running_loss / len(loader)
#     acc = 100 * correct / total
#     return avg_loss, acc



# # NO marks
# def train_model(model,train_loader,val_loader,criterion,optimizer,num_epochs,device,max_atom):
#     model.to(device)
#     train_losses = []
#     val_losses = []
#     train_accs = []
#     val_accs = []
#
#
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#
#         # Training phase
#         for inputs, labels, smi_length,smiles in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
#
#             batch_size = inputs.size(0)
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             # print('outputs_before',outputs)
#             # mask_outputs = []
#             # for i in range(batch_size):
#             #     smiles_single= smiles[i]
#             #     smiles_mask=atom_flags_padded(smiles_single)
#             #     mask_outputs.append(smiles_mask)
#             # mask_outputs=torch.tensor(mask_outputs,device=outputs.device)
#             # outputs_refine = outputs * mask_outputs  # 后n个样本的loss被置零
#             # A outputs
#             loss = criterion(outputs, labels)
#             loss=loss.mean()
#             # print('outputs_mask',outputs_refine)
#
#             # 将后 (batch_size - n) 个样本的损失置零
#             # mask = torch.zeros_like(loss)
#             # for i in range(batch_size):
#             #     k = smi_length[i]
#             #     mask[i, :k] = 1
#             # loss = loss * mask  # 后n个样本的loss被置零
#             # # 对未被掩码的loss求平均（可选）
#             # loss = loss.sum() / smi_length.sum()
#
#             loss.backward()
#             optimizer.step()
#             # loss = loss * mask  # 后n个样本的loss被置零
#             # # 对未被掩码的loss求平均（可选）
#             # loss = loss.sum() / smi_length.sum()
#             running_loss += loss.item()
#
#             threshold = 0.5
#             # print(outputs)
#             outputs = torch.sigmoid(outputs)
#             predicted = (outputs >= threshold).float()
#             for i in range(labels.size(0)):
#                 label_atoms=labels[i][:smi_length[i]]
#                 predicted_atoms=predicted[i][:smi_length[i]]
#                 # print(smiles)
#                 # print(label_atoms)
#                 # print(predicted_atoms)
#
#                 atom_error=get_atom_error(label_atoms,predicted_atoms)
#                 if 0<=atom_error<=max_atom:
#                 # if torch.equal(label_atoms, predicted_atoms):
#                     correct += 1  # Count as correct
#                 total += labels.size(0)
#
#         train_loss = running_loss / len(train_loader)
#         train_acc = 100 * correct / total
#         train_losses.append(train_loss)
#         train_accs.append(train_acc)
#
#         # Validation phase
#         val_loss, val_acc = evaluate_model(model, val_loader, criterion,device,max_atom)
#         val_losses.append(val_loss)
#         val_accs.append(val_acc)
#
#         print(f'Epoch {epoch + 1}/{num_epochs} - '
#               f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
#               f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
#
#     return model,train_losses,val_losses,train_accs,val_accs
#
# def evaluate_model(model, loader, criterion,device,max_atom):
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         for inputs, labels, smi_length,smiles in loader:
#             batch_size = inputs.size(0)
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             A=outputs
#
#             loss = criterion(outputs, labels)
#
#             # 将后 (batch_size - n) 个样本的损失置零
#             mask = torch.zeros_like(loss)
#             for i in range(batch_size):
#                 k = smi_length[i]
#                 mask[i, :k] = 1
#             loss = loss * mask  # 后n个样本的loss被置零
#             # 对未被掩码的loss求平均（可选）
#             loss = loss.sum() / smi_length.sum()
#
#             running_loss += loss.item()
#
#
#             threshold = 0.5
#             outputs = torch.sigmoid(outputs)
#             predicted = (outputs >= threshold).float()
#             for i in range(labels.size(0)):
#                 label_atoms = labels[i][:smi_length[i]]
#                 predicted_atoms = predicted[i][:smi_length[i]]
#                 atom_error=get_atom_error(label_atoms,predicted_atoms)
#                 if 0<=atom_error<=max_atom:
#                 # if torch.equal(label_atoms, predicted_atoms):
#                     correct += 1  # Count as correct
#                 total += labels.size(0)
#
#     loss = running_loss / len(loader)
#     acc = 100 * correct / total
#     return loss, acc
#

# # train ,valid
# def train_model(model,train_loader,val_loader,criterion,optimizer,num_epochs,device,max_atom):
#     model.to(device)
#     train_losses = []
#     val_losses = []
#     train_accs = []
#     val_accs = []
#     best_val_acc = 0.0
#     best_model_state = None
#
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#
#         # Training phase
#         for inputs, labels, smi_length,smiles in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
#
#             batch_size = inputs.size(0)
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             print('outputs_before',outputs)
#             mask_outputs = []
#             for i in range(batch_size):
#                 smiles_single= smiles[i]
#                 smiles_mask=atom_flags_padded(smiles_single)
#                 mask_outputs.append(smiles_mask)
#             mask_outputs=torch.tensor(mask_outputs,device=outputs.device)
#             outputs_refine = outputs * mask_outputs  # 后n个样本的loss被置零
#             # A= outputs
#             loss = criterion(outputs_refine, labels)
#             # print('outputs_refine',outputs_refine)
#             # print('labels',labels)
#
#             # 将后 (batch_size - n) 个样本的损失置零
#             mask = torch.zeros_like(loss)
#             for i in range(batch_size):
#                 k = smi_length[i]
#                 mask[i, :k] = 1
#             loss = loss * mask  # 后n个样本的loss被置零
#
#             # 对未被掩码的loss求平均（可选）
#             loss = loss.sum() / smi_length.sum()
#
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#
#             threshold = 0.5
#             # print(outputs)
#             outputs = torch.sigmoid(outputs)
#             predicted = (outputs>= threshold).float()
#
#             for i in range(labels.size(0)):
#                 label_atoms=labels[i][:smi_length[i]]
#                 predicted_atoms=predicted[i][:smi_length[i]]
#                 # print(smiles)
#                 # print(label_atoms)
#                 # print(predicted_atoms)
#
#                 atom_error=get_atom_error(label_atoms,predicted_atoms)
#                 if 0<=atom_error<=max_atom:
#                 # if torch.equal(label_atoms, predicted_atoms):
#                     correct += 1  # Count as correct
#                 total += labels.size(0)
#
#         train_loss = running_loss / len(train_loader)
#         train_acc = 100 * correct / total
#         train_losses.append(train_loss)
#         train_accs.append(train_acc)
#
#         # Validation phase
#         val_loss, val_acc = evaluate_model(model, val_loader, criterion,device,max_atom)
#         val_losses.append(val_loss)
#         val_accs.append(val_acc)
#
#         print(f'Epoch {epoch + 1}/{num_epochs} - '
#               f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
#               f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
#         # Save the best model based on validation accuracy
#
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_model_state = model.state_dict()
#             print(f'✅ New best model found at Epoch {epoch + 1} with Val Acc: {val_acc:.2f}%')
#
#
#     return best_model_state,best_val_acc,train_losses,val_losses,train_accs,val_accs
#
#
# def evaluate_model(model, loader, criterion,device,max_atom):
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         for inputs, labels, smi_length,smiles in loader:
#             batch_size = inputs.size(0)
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             # print('smile',smiles)
#             # print('outputs', outputs)
#             # print('outputs',outputs.size())
#
#             mask_outputs = []
#             for i in range(batch_size):
#                 smiles_single= smiles[i]
#                 smiles_mask=atom_flags_padded(smiles_single)
#                 mask_outputs.append(smiles_mask)
#
#             mask_outputs=torch.tensor(mask_outputs,device=outputs.device)
#             # print('mask_outputs',mask_outputs)
#             outputs_refine = outputs * mask_outputs  # 后n个样本的loss被置零
#             # print('outputs_refine',outputs_refine)
#             # print('outputs_refine', outputs_refine.size())
#             # print('labels',labels)
#
#             loss = criterion(outputs_refine, labels)
#             # print('loss', loss)
#
#
#             # 将后 (batch_size - n) 个样本的损失置零
#             mask = torch.zeros_like(loss)
#             for i in range(batch_size):
#                 k = smi_length[i]
#                 mask[i, :k] = 1
#             # print('mark',mask)
#             loss = loss * mask  # 后n个样本的loss被置零
#             # print('loss_mask',loss)
#             # 对未被掩码的loss求平均（可选）
#             loss = loss.sum() / smi_length.sum()
#             # print('loss_sum', loss)
#             running_loss += loss.item()
#             # print('running_loss',running_loss)
#             threshold = 0.5
#             outputs = torch.sigmoid(outputs)
#             predicted = (outputs >= threshold).float()
#             # print('predicted', predicted)
#             for i in range(labels.size(0)):
#                 label_atoms = labels[i][:smi_length[i]]
#                 predicted_atoms = predicted[i][:smi_length[i]]
#                 atom_error=get_atom_error(label_atoms,predicted_atoms)
#                 if 0<=atom_error<=max_atom:
#                 # if torch.equal(label_atoms, predicted_atoms):
#                     correct += 1  # Count as correct
#                 total += labels.size(0)
#
#     loss = running_loss / len(loader)
#     acc = 100 * correct / total
#     return loss, acc

# ### add mcc
#

#
# # add test + MCC
# def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, device, max_atom):
#     model.to(device)
#     train_losses, val_losses = [], []
#     train_accs, val_accs = [], []
#     train_cacc, val_cacc = [], []
#     train_mccs, val_mccs = [], []
#
#     best_val_acc = 0.0
#     best_model_state = None
#
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct, total = 0, 0
#         all_labels, all_preds = [], []
#
#         for inputs, labels, smi_length, smiles in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
#             batch_size = inputs.size(0)
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#
#             # Construct attention mask for output
#             mask_outputs = []
#             for i in range(batch_size):
#                 smiles_single = smiles[i]
#                 smiles_mask = atom_flags_padded(smiles_single)
#                 mask_outputs.append(smiles_mask)
#             mask_outputs = torch.tensor(mask_outputs, device=outputs.device)
#
#             # Apply mask to the output
#             outputs_refine = outputs * mask_outputs
#
#             # Compute loss with masking
#             loss = criterion(outputs_refine, labels)
#             mask = torch.zeros_like(loss)
#             for i in range(batch_size):
#                 k = smi_length[i]
#                 mask[i, :k] = 1
#             loss = loss * mask
#             loss = loss.sum() / smi_length.sum()
#
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#
#             # Compute training accuracy and MCC
#             outputs_sigmoid = torch.sigmoid(outputs)
#             predicted = (outputs_sigmoid >= 0.5).float()
#             for i in range(labels.size(0)):
#                 label_atoms = labels[i][:smi_length[i]].cpu().numpy()
#                 predicted_atoms = predicted[i][:smi_length[i]].cpu().numpy()
#                 atom_error = get_atom_error(torch.tensor(label_atoms), torch.tensor(predicted_atoms))
#                 if 0 <= atom_error <= max_atom:
#                     correct += 1
#                 total += 1
#                 all_labels.extend(label_atoms)
#                 all_preds.extend(predicted_atoms)
#
#         train_loss = running_loss / len(train_loader)
#         train_acc = 100 * correct / total
#         train_mcc = matthews_corrcoef(all_labels, all_preds)
#
#         train_losses.append(train_loss)
#         train_accs.append(train_acc)
#         train_mccs.append(train_mcc)
#
#         # Validation
#         val_loss, val_acc, val_mcc = evaluate_model(model, val_loader, criterion, device, max_atom)
#         val_losses.append(val_loss)
#         val_accs.append(val_acc)
#         val_mccs.append(val_mcc)
#
#         print(f'Epoch {epoch + 1}/{num_epochs} - '
#               f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train MCC: {train_mcc:.4f} | '
#               f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val MCC: {val_mcc:.4f}')
#
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_model_state = model.state_dict()
#             print(f'✅ New best model found at Epoch {epoch + 1} with Val Acc: {val_acc:.2f}%')
#
#     # Final evaluation on test set
#     test_loss, test_acc, test_mcc = evaluate_model(model, test_loader, criterion, device, max_atom)
#     print(f'\n Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Test MCC: {test_mcc:.4f}')
#
#     if best_model_state is not None:
#         model.load_state_dict(best_model_state)
#         torch.save(model.state_dict(), 'best_model.pth')
#         print(' Best model saved as best_model.pth')
#         test_loss_best, test_acc_best, test_mcc_best = evaluate_model(model, test_loader, criterion, device, max_atom)
#         print(f'\n Test Loss best: {test_loss_best:.4f}, Test Acc: {test_acc_best:.2f}%, Test MCC best: {test_mcc_best:.4f}')
#
#     return best_model_state, best_val_acc, train_losses, val_losses, train_accs, val_accs, train_mccs, val_mccs, test_loss, test_acc, test_mcc
#
# #
# def evaluate_model(model, loader, criterion, device, max_atom):
#     model.eval()
#     running_loss = 0.0
#     correct, total = 0, 0
#     all_labels, all_preds = [], []
#
#     with torch.no_grad():
#         for inputs, labels, smi_length, smiles in loader:
#             batch_size = inputs.size(0)
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#
#             loss = criterion(outputs, labels)
#             mask = torch.zeros_like(loss)
#             for i in range(batch_size):
#                 k = smi_length[i]
#                 mask[i, :k] = 1
#             loss = loss * mask
#             loss = loss.sum() / smi_length.sum()
#             running_loss += loss.item()
#
#             outputs_sigmoid = torch.sigmoid(outputs)
#             predicted = (outputs_sigmoid >= 0.5).float()
#
#             for i in range(labels.size(0)):
#                 label_atoms = labels[i][:smi_length[i]].cpu().numpy()
#                 predicted_atoms = predicted[i][:smi_length[i]].cpu().numpy()
#                 atom_error = get_atom_error(torch.tensor(label_atoms), torch.tensor(predicted_atoms))
#                 if 0 <= atom_error <= max_atom:
#                     correct += 1
#                 total += 1
#                 all_labels.extend(label_atoms)
#                 all_preds.extend(predicted_atoms)
#
#     avg_loss = running_loss / len(loader)
#     acc = 100 * correct / total
#     mcc = matthews_corrcoef(all_labels, all_preds)
#     return avg_loss, acc, mcc


## add cacc (direct acc)

from sklearn.metrics import matthews_corrcoef
import torch
from tqdm import tqdm

# add test + MCC + direct ACC (cacc)
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, device, max_atom):
    model.to(device)
    train_losses, val_losses = [], []
    train_daccs, val_daccs = [], []         # 原先的 atom_error-based ACC
    train_saccs, val_saccs = [], []       # ✅ 新增：直接计算的逐元素ACC
    train_mccs, val_mccs = [], []

    best_val_dacc = 0.0
    best_val_mcc = 0.0

    best_model_state_mcc=None
    best_model_state_dacc=None
    best_model_state_sacc=None
    best_val_sacc=0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        scorrect, stotal = 0, 0           # ✅ 新增：直接ACC计数
        all_labels, all_preds = [], []

        for inputs, labels, smi_length, smiles in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            batch_size = inputs.size(0)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # Construct attention mask for output
            mask_outputs = []
            for i in range(batch_size):
                smiles_single = smiles[i]
                smiles_mask = atom_flags_padded(smiles_single)
                mask_outputs.append(smiles_mask)
            mask_outputs = torch.tensor(mask_outputs, device=outputs.device)

            # Apply mask to the output
            outputs_refine = outputs * mask_outputs

            # Compute loss with masking
            loss = criterion(outputs_refine, labels)
            mask = torch.zeros_like(loss)
            for i in range(batch_size):
                k = smi_length[i]
                mask[i, :k] = 1
            loss = loss * mask
            loss = loss.sum() / smi_length.sum()

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Compute training accuracy and MCC
            outputs_sigmoid = torch.sigmoid(outputs)
            predicted = (outputs_sigmoid >= 0.5).float()

            for i in range(labels.size(0)):
                label_atoms = labels[i][:smi_length[i]].cpu().numpy()
                predicted_atoms = predicted[i][:smi_length[i]].cpu().numpy()

                # 原 ACC (通过 atom_error 判定)
                atom_error = get_atom_error(torch.tensor(label_atoms), torch.tensor(predicted_atoms))
                if 0 <= atom_error <= max_atom:
                    correct += 1
                total += 1

                # ✅ 新增：直接计算每个atom的匹配正确率
                scorrect += (label_atoms == predicted_atoms).sum()
                stotal += len(label_atoms)

                all_labels.extend(label_atoms)
                all_preds.extend(predicted_atoms)

        train_loss = running_loss / len(train_loader)
        train_dacc = 100 * correct / total
        train_sacc = 100 * scorrect / stotal       # ✅ 新增：直接逐元素 ACC
        train_mcc = matthews_corrcoef(all_labels, all_preds)

        train_losses.append(train_loss)
        train_daccs.append(train_dacc)
        train_saccs.append(train_sacc)             # ✅ 新增
        train_mccs.append(train_mcc)

        # Validation
        val_loss, val_dacc, val_sacc, val_mcc = evaluate_model(model, val_loader, criterion, device, max_atom)
        val_losses.append(val_loss)
        val_daccs.append(val_dacc)
        val_saccs.append(val_sacc)                 # ✅ 新增
        val_mccs.append(val_mcc)

        print(f'Epoch {epoch + 1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Dacc: {train_dacc:.2f}%, '
              f'Train CAcc: {train_sacc:.2f}%, Train MCC: {train_mcc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Val Dacc: {val_dacc:.2f}%, '
              f'Val SAcc: {val_sacc:.2f}%, Val MCC: {val_mcc:.4f}')

        if val_dacc > best_val_dacc:
            best_val_dacc = val_dacc
            best_model_state_dacc = model.state_dict()
            print(f'✅ New best model found at Epoch {epoch + 1} with Val DAcc: {val_dacc:.2f}%')

        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_model_state_mcc = model.state_dict()
            print(f'✅ New best model found at Epoch {epoch + 1} with Val Mcc: {val_mcc:.2f}')

            best_val_sacc = val_sacc
            best_model_state_sacc = model.state_dict()
            print(f'✅ New best model found at Epoch {epoch + 1} with Val SAcc: {val_sacc:.2f}')

        # if val_sacc > best_val_sacc:
        #     best_val_sacc = val_sacc
        #     best_model_state_sacc = model.state_dict()
        #     print(f'✅ New best model found at Epoch {epoch + 1} with Val SAcc: {val_sacc:.2f}')



    # Final evaluation on test set
    test_loss, test_dacc, test_sacc, test_mcc = evaluate_model(model, test_loader, criterion, device, max_atom)
    print(f'\n Test Loss: {test_loss:.4f}, Test Acc(atom): {test_dacc:.2f}%, '
          f'Test SAcc: {test_sacc:.2f}%, Test MCC: {test_mcc:.4f}')

    # if best_model_state_mcc is not None:
    #     model.load_state_dict(best_model_state_mcc)
    #     # torch.save(model.state_dict(), 'best_model_state_mcc.pth')
    #     print(' Best model saved as best_model_state_mcc.pth')
    #     test_loss_best, test_acc_best, test_sacc_best, test_mcc_best = evaluate_model(model, test_loader, criterion, device, max_atom)
    #     print(f'\n Test(best) Loss: {test_loss_best:.4f}, Test Acc(atom): {test_acc_best:.2f}%, '
    #           f'Test CAcc: {test_sacc_best:.2f}%, Test MCC: {test_mcc_best:.4f}')

    return (best_model_state_dacc, best_model_state_mcc,best_model_state_sacc, best_val_dacc,best_val_mcc,best_val_sacc, train_losses,
            val_losses,train_daccs, val_daccs, train_saccs, val_saccs, train_mccs, val_mccs,test_loss, test_dacc, test_sacc, test_mcc)


def evaluate_model(model, loader, criterion, device, max_atom):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    scorrect, stotal = 0, 0            # ✅ 新增：直接ACC
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels, smi_length, smiles in loader:
            batch_size = inputs.size(0)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            mask = torch.zeros_like(loss)
            for i in range(batch_size):
                k = smi_length[i]
                mask[i, :k] = 1
            loss = loss * mask
            loss = loss.sum() / smi_length.sum()
            running_loss += loss.item()

            outputs_sigmoid = torch.sigmoid(outputs)
            predicted = (outputs_sigmoid >= 0.5).float()

            for i in range(labels.size(0)):
                label_atoms = labels[i][:smi_length[i]].cpu().numpy()
                predicted_atoms = predicted[i][:smi_length[i]].cpu().numpy()
                atom_error = get_atom_error(torch.tensor(label_atoms), torch.tensor(predicted_atoms))
                if 0 <= atom_error <= max_atom:
                    correct += 1
                total += 1

                # ✅ 新增：直接ACC
                scorrect += (label_atoms == predicted_atoms).sum()
                stotal += len(label_atoms)

                all_labels.extend(label_atoms)
                all_preds.extend(predicted_atoms)

    avg_loss = running_loss / len(loader)
    dacc = 100 * correct / total
    sacc = 100 * scorrect / stotal          # ✅ 新增
    mcc = matthews_corrcoef(all_labels, all_preds)
    return avg_loss, dacc, sacc, mcc