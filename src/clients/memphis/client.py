"""
Adapted from NVFlare "Hello PyTorch" (hello-pt) example code.

Simulates the San Diego Hospital as outlined in the Methods.
"""
import os 

import torch 
from torch import nn 

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter

from server.model import SimpleNetwork
from clients.datasets import ClinicalDataset, RNADataset

def main():
    DATASET_PATH="/Users/tyleryang/Developer/CMU-NVIDIA-Hackathon/rna-cd-data/"
    # ----------------------------------------------------------------------- #
    # Client-Specific Constants
    # ----------------------------------------------------------------------- #
    IDS = ['3B_208', '3B_217', '3B_225', '3B_227', '3B_229', '3B_230', '3B_250', '3B_262', '3B_266', '3B_267', '3B_277', '3B_281', '3B_288', '3B_292', '3B_302', '3B_303', '3B_304', '3B_309', '3B_310', '3B_319', '3B_321', '3B_322', '3B_328', '3B_337', '3B_338', '3B_342', '3B_351', '3B_354', '3B_357', '3B_361', '3B_362', '3B_365', '3B_367', '3B_370', '3B_385', '3B_389', '3B_390', '3B_397', '3B_399', '3B_408', '3B_410', '3B_411', '3B_413', '3B_415', '3B_417', '3B_418', '3B_426', '3B_428', '3B_429', '3B_431']
    assert len(IDS) == 50, f"expected 50 IDS in Memphis (CHIMERA Task 3 Cohort B), got {len(IDS)}"

    # ----------------------------------------------------------------------- #
    # Initialize Model
    # ----------------------------------------------------------------------- #

    # Shared across all clients
    model = SimpleNetwork(rna_dim=19359, clinical_dim=13)

    # Free for Client to modify
    config = {
        "batch_size": 5,
        "epochs": 5,
        "learning_rate": 1e-3
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # ----------------------------------------------------------------------- #
    # Load Client-Specific Data for Federated Training
    # ----------------------------------------------------------------------- #
    clinical_files = [
        os.path.join(DATASET_PATH, f"{id}/{id}_CD.json") for id in IDS
    ]  # Add all clinical file paths here
    clinical_ds = ClinicalDataset(clinical_files)

    CDloader = torch.utils.data.DataLoader(
        clinical_ds, 
        batch_size=config['batch_size'], 
        shuffle=True
    )

    # ----------------------------------------------------------------------- #
    # Load Evaluation Data (only b/c we are simulating FL)
    # ----------------------------------------------------------------------- #
    def evaluate(net, data_loader, device):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in data_loader:
                # (optional) use GPU to speed things up
                clinical, rnaseq, labels = data.to(device)
                outputs = net(clinical, rnaseq)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f"Accuracy of the network on the 26 test subjects from Cohort A: {100 * correct // total} %")
        return 100 * correct // total
    
    TEST_IDS = ['3A_137', '3A_138', '3A_139', '3A_140', '3A_141', '3A_142', '3A_143', '3A_144', '3A_145', '3A_146', '3A_147', '3A_148', '3A_149', '3A_153', '3A_154', '3A_157', '3A_158', '3A_160', '3A_162', '3A_163', '3A_165', '3A_168', '3A_169', '3A_186', '3A_190', '3A_191']
    assert len(TEST_IDS) == 26, f"expected 26 IDS for evaluation, got {len(TEST_IDS)}"


    # ----------------------------------------------------------------------- #
    # NVFlare Initialization
    # ----------------------------------------------------------------------- #
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]
    summary_writer = SummaryWriter()
    print(f"San Diego Hospital corresponds to site_name: {client_name}")

    # ----------------------------------------------------------------------- #
    # NVFlare Training Loop
    # ----------------------------------------------------------------------- #
    while flare.is_running():
        global_model = flare.receive()
        print(f"San Diego Hospital site_name: {client_name}")
        print(f"current_round={global_model.current_round}") # type: ignore

        model.load_state_dict(global_model.params) # type: ignore
        model.to(device)

        # evaluate on received model
        accuracy = evaluate(model, test_loader, device)

        steps = config['epochs'] * len(train_loader)
        for epoch in range(config['epochs']):
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                images, labels = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()

                predictions = model(images)
                cost = loss(predictions, labels)
                cost.backward()
                optimizer.step()

                running_loss += cost.item()
                if i % 5 == 4:
                    avg_loss = running_loss / 5
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}")

                    # Optional: Log metrics
                    global_step = global_model.current_round * steps + epoch * len(train_loader) + i # type: ignore
                    summary_writer.add_scalar(tag="loss", scalar=avg_loss, global_step=global_step)

                    print(f"site={client_name}, Epoch: {epoch}/{config['epochs']}, Iteration: {i}, Loss: {running_loss}")
                    running_loss = 0.0

        print(f"Finished Training for San Diego Hospital, site_name: {client_name}")

        PATH = "./san-diego.pth"
        torch.save(model.state_dict(), PATH)

        # (7) construct trained FL model
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        print(f"site: {client_name}, sending model to server.")
        # (8) send model back to NVFlare
        flare.send(output_model)

if __name__ == "__main__":
    main()