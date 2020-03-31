import torch
import os

from fedl.users.useravg import UserAVG
from fedl.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data

class FedAvg(Server):
    def __init__(self, dataset, model, batch_size, learning_rate,meta_learning_rate, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users):
        super().__init__(dataset, model[0], batch_size, learning_rate, meta_learning_rate, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users)

        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserAVG(id, train, test, model, batch_size, learning_rate,meta_learning_rate,lamda, local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Finished creating server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            loss_ = 0
            self.send_parameters()
            self.selected_users = self.select_users(glob_iter,self.num_users)
            for user in self.selected_users:
                loss_ += user.train(self.local_epochs) * user.train_samples
            self.aggregate_parameters()
            loss_ /= self.total_train_samples
            loss.append(loss_)
            print(loss_)
        print(loss)
        self.save_model()