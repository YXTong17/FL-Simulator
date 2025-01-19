import torch
from client import *
from .server import Server


class FedFGAC(Server):
    def __init__(
        self, device, model_func, init_model, init_par_list, datasets, method, args
    ):
        super().__init__(
            device, model_func, init_model, init_par_list, datasets, method, args
        )
        # rebuild
        self.comm_vecs = {
            "Params_list": init_par_list.clone().detach(),
        }
        self.Client = fedfgac

        self.clients_label_counts = torch.zeros((args.total_client, init_model.n_cls))
        self.classifier_shape = list(init_model.model.fc.parameters())[0].shape
        self.classifier_weight_size = (
            self.classifier_shape[0] * self.classifier_shape[1]
        )
        self.classifier_bais_size = self.classifier_shape[0]

    def process_for_communication(self, client, Averaged_update):
        self.comm_vecs["Params_list"].copy_(self.server_model_params_list)

    def postprocess(self, client, received_vecs):
        self.clients_label_counts[client] = received_vecs["local_label_counts"]

    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        server_model_params_list = (
            self.server_model_params_list
            + self.args.global_learning_rate * Averaged_update
        )

        client_label_counts = self.clients_label_counts[selected_clients]
        total_label_counts = client_label_counts.sum(dim=0)
        classifier_weights = client_label_counts / total_label_counts.clamp(min=1e-6)

        server_classifier_weight_params = self.server_model_params_list[
            -(
                self.classifier_weight_size + self.classifier_bais_size
            ) : -self.classifier_bais_size
        ].reshape(self.classifier_shape)
        server_classifier_bais_params = self.server_model_params_list[
            -self.classifier_bais_size :
        ]
        client_classifier_weight_updates = [
            p[
                -(
                    self.classifier_weight_size + self.classifier_bais_size
                ) : -self.classifier_bais_size
            ].reshape(self.classifier_shape)
            for p in self.clients_updated_params_list[selected_clients]
        ]
        client_classifier_bais_updates = [
            p[-self.classifier_bais_size :]
            for p in self.clients_updated_params_list[selected_clients]
        ]

        for client in range(len(selected_clients)):
            # server_classifier_weight_params += (
            #     classifier_weights[client] @ client_classifier_weight_updates[client]
            # )
            # server_classifier_bais_params += (
            #     classifier_weights[client] * client_classifier_bais_updates[client]
            # )
            for class_idx, c_w in enumerate(classifier_weights[client]):
                server_classifier_weight_params[class_idx] += (
                    client_classifier_weight_updates[client][class_idx] * c_w
                )
                server_classifier_bais_params[class_idx] += (
                    client_classifier_bais_updates[client][class_idx] * c_w
                )

        server_model_params_list[
            -(
                self.classifier_weight_size + self.classifier_bais_size
            ) : -self.classifier_bais_size
        ] = server_classifier_weight_params.reshape(-1)
        server_model_params_list[-self.classifier_bais_size :] = (
            server_classifier_bais_params
        )

        return server_model_params_list
