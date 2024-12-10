import torch
import pandas as pd

class Rule_based_Metric():

    def rule_based_ade(rule_based_result, gt):
        rule_based_result = torch.tensor(rule_based_result, dtype=torch.float, device=gt.device)
        ade = torch.norm(rule_based_result[..., :2] - gt[0][..., :2], p=2, dim=-1).mean(-1)
        min_ade = ade.min(-1)[0]
        return min_ade
    
    def rule_based_fde(rule_based_result, gt):
        rule_based_result = torch.tensor(rule_based_result, dtype=torch.float, device=gt.device)
        fde = torch.norm(rule_based_result[-1, :2] - gt[0][-1, :2], p=2, dim=-1).mean(-1)
        min_fde = fde.min(-1)[0]
        return min_fde
    
    def model_ade(model_result, gt):
        ade = torch.norm(model_result[0][..., :2] - gt[0][..., :2], p=2, dim=-1).mean(-1)
        min_ade = ade.min(-1)[0]
        return min_ade
    
    def model_fde(model_result, gt):
        fde = torch.norm(model_result[0][-1, :2] - gt[0][-1, :2], p=2, dim=-1).mean(-1)
        min_fde = fde.min(-1)[0]
        return min_fde
    
    def save_metric(trainer, batch_idx, gt, model, cv, ca, ctrv, ctra, val, save_dir):
        epoch = trainer.current_epoch
        
        model_min_ade = Rule_based_Metric.model_ade(model, gt)
        model_min_fde = Rule_based_Metric.model_fde(model, gt)
        cv_min_ade = Rule_based_Metric.rule_based_ade(cv, gt)
        cv_min_fde = Rule_based_Metric.rule_based_fde(cv, gt)
        ca_min_ade = Rule_based_Metric.rule_based_ade(ca, gt)
        ca_min_fde = Rule_based_Metric.rule_based_fde(ca, gt)
        ctrv_min_ade = Rule_based_Metric.rule_based_ade(ctrv, gt)
        ctrv_min_fde = Rule_based_Metric.rule_based_fde(ctrv, gt)
        ctra_min_ade = Rule_based_Metric.rule_based_ade(ctra, gt)
        ctra_min_fde = Rule_based_Metric.rule_based_fde(ctra, gt)
        
        # f = open("./result_test/metric.txt", "a+")
        # if val == True:
        #     f.write("Validation\n")
        #     f.write("Epoch: {}, Batch Index: {}\n".format(epoch, batch_idx))
        # else:
        #     f.write("Training\n")
        #     f.write("Epoch: {}, Batch Index: {}\n".format(epoch, batch_idx))
        # f.write("Model minADE: {}, Model minFDE: {}\n".format(model_min_ade, model_min_fde))
        # f.write("CV minADE: {}, CV minFDE: {}\n".format(cv_min_ade, cv_min_fde))
        # f.write("CA minADE: {}, CA minFDE: {}\n".format(ca_min_ade, ca_min_fde))
        # f.write("CTRV minADE: {}, CTRV minFDE: {}\n".format(ctrv_min_ade, ctrv_min_fde))
        # f.write("CTRA minADE: {}, CTRA minFDE: {}\n\n".format(ctra_min_ade, ctra_min_fde))
        # f.close()
        
        # save metric to csv file column name (epoch, batch_idx, model_min_ade, model_min_fde, cv_min_ade, cv_min_fde, ca_min_ade, ca_min_fde, ctrv_min_ade, ctrv_min_fde, ctra_min_ade, ctra_min_fde)
        if val == 0: 
            metric = pd.DataFrame([["Training", epoch, batch_idx, model_min_ade.item(), model_min_fde.item(), cv_min_ade.item(), cv_min_fde.item(), ca_min_ade.item(), ca_min_fde.item(), ctrv_min_ade.item(), ctrv_min_fde.item(), ctra_min_ade.item(), ctra_min_fde.item()]], columns=["process", "epoch", "batch_idx", "model_min_ade", "model_min_fde", "cv_min_ade", "cv_min_fde", "ca_min_ade", "ca_min_fde", "ctrv_min_ade", "ctrv_min_fde", "ctra_min_ade", "ctra_min_fde"])
        elif val == 1:
            metric = pd.DataFrame([["Validation", epoch, batch_idx, model_min_ade.item(), model_min_fde.item(), cv_min_ade.item(), cv_min_fde.item(), ca_min_ade.item(), ca_min_fde.item(), ctrv_min_ade.item(), ctrv_min_fde.item(), ctra_min_ade.item(), ctra_min_fde.item()]], columns=["process", "epoch", "batch_idx", "model_min_ade", "model_min_fde", "cv_min_ade", "cv_min_fde", "ca_min_ade", "ca_min_fde", "ctrv_min_ade", "ctrv_min_fde", "ctra_min_ade", "ctra_min_fde"])
        else:
            metric = pd.DataFrame([["Test", epoch, batch_idx, model_min_ade.item(), model_min_fde.item(), cv_min_ade.item(), cv_min_fde.item(), ca_min_ade.item(), ca_min_fde.item(), ctrv_min_ade.item(), ctrv_min_fde.item(), ctra_min_ade.item(), ctra_min_fde.item()]], columns=["process", "epoch", "batch_idx", "model_min_ade", "model_min_fde", "cv_min_ade", "cv_min_fde", "ca_min_ade", "ca_min_fde", "ctrv_min_ade", "ctrv_min_fde", "ctra_min_ade", "ctra_min_fde"])
        
        if val != 1 and epoch == 0 and batch_idx == 0:
            metric.to_csv(save_dir + "metric.csv", header= True, index=False)
            # metric.to_csv("./eval_result_batch_8_gpus_1_csv/metric.csv", header= True, index=False)
        else:
            metric.to_csv(save_dir + "metric.csv", mode='a', header=False, index=False)
            # metric.to_csv("./eval_result_batch_8_gpus_1_csv/metric.csv", mode='a', header=False, index=False)
        
        
        
        
        