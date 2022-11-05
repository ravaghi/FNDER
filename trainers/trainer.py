from tqdm import tqdm
from sklearn import metrics
import torch
import wandb

from models.chordmixer import ChordMixer


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, device, criterion, optimizer):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, current_epoch_nr):
        self.model.train()

        num_batches = len(self.train_dataloader)

        preds = []
        targets = []
        correct = 0
        running_loss = 0.0
        items_processed = 0

        loop = tqdm(enumerate(self.train_dataloader), total=num_batches)
        for idx, (x, y, seq_len, bucket) in loop:
            x = x.to(self.device)
            y = y.to(self.device)

            if isinstance(self.model, ChordMixer):
                y_hat = self.model(x, seq_len)
            else:
                y_hat = self.model(x)

            loss = self.criterion(y_hat, y)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()

            _, predicted = y_hat.max(1)
            items_processed += y.size(0)
            correct += predicted.eq(y).sum().item()

            targets.extend(y.detach().cpu().numpy().flatten())
            preds.extend(predicted.detach().cpu().numpy().flatten())

            loop.set_description(f'Epoch {current_epoch_nr + 1}')
            loop.set_postfix(train_acc=round(correct / items_processed, 4),
                             train_loss=round(running_loss / items_processed, 4))

        train_auc = metrics.roc_auc_score(targets, preds)
        train_accuracy = correct / items_processed
        train_loss = running_loss / items_processed
        wandb.log({'train_loss': train_loss}, step=current_epoch_nr)
        wandb.log({'train_accuracy': train_accuracy}, step=current_epoch_nr)
        wandb.log({'train_auc': train_auc}, step=current_epoch_nr)

    def evaluate(self, current_epoch_nr, scheduler):
        self.model.eval()

        num_batches = len(self.val_dataloader)

        preds = []
        targets = []
        correct = 0
        running_loss = 0.0
        items_processed = 0

        with torch.no_grad():
            loop = tqdm(enumerate(self.val_dataloader), total=num_batches)
            for idx, (x, y, seq_len, bucket) in loop:
                x = x.to(self.device)
                y = y.to(self.device)

                if isinstance(self.model, ChordMixer):
                    y_hat = self.model(x, seq_len)
                else:
                    y_hat = self.model(x)

                loss = self.criterion(y_hat, y)

                running_loss += loss.item()

                _, predicted = y_hat.max(1)
                items_processed += y.size(0)
                correct += predicted.eq(y).sum().item()

                targets.extend(y.detach().cpu().numpy().flatten())
                preds.extend(predicted.detach().cpu().numpy().flatten())

                loop.set_description(f'Epoch {current_epoch_nr + 1}')
                loop.set_postfix(val_acc=round(correct / items_processed, 4),
                                 val_loss=round(running_loss / items_processed, 4))

            val_auc = metrics.roc_auc_score(targets, preds)
            validation_accuracy = correct / items_processed
            validation_loss = running_loss / num_batches
            wandb.log({'val_loss': validation_loss}, step=current_epoch_nr)
            wandb.log({'val_accuracy': validation_accuracy}, step=current_epoch_nr)
            wandb.log({'val_auc': val_auc}, step=current_epoch_nr)

        scheduler.step(validation_accuracy)

    def test(self):
        self.model.eval()

        num_batches = len(self.test_dataloader)

        preds = []
        targets = []
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            loop = tqdm(enumerate(self.test_dataloader), total=num_batches)
            for idx, (x, y, seq_len, bucket) in loop:
                x = x.to(self.device)
                y = y.to(self.device)

                if isinstance(self.model, ChordMixer):
                    y_hat = self.model(x, seq_len)
                else:
                    y_hat = self.model(x)

                loss = self.criterion(y_hat, y)

                running_loss += loss.item()

                _, predicted = y_hat.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                targets.extend(y.detach().cpu().numpy().flatten())
                preds.extend(predicted.detach().cpu().numpy().flatten())

                loop.set_description('Testing')
                loop.set_postfix(test_acc=round(correct / total, 4),
                                 test_loss=round(running_loss / total, 4))

        test_auc = metrics.roc_auc_score(targets, preds)
        test_accuracy = correct / total
        test_loss = running_loss / num_batches
        wandb.run.summary["test_loss"] = test_loss
        wandb.run.summary["test_accuracy"] = test_accuracy
        wandb.run.summary["test_auc"] = test_auc
