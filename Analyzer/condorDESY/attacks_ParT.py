from definitions_ParT import integer_variables_by_candidate, cands_per_variable, vars_per_candidate
import torch
import numpy as np
from helpers_advertorch import *

def apply_noise(sample, magn=1e-2,offset=[0], dev=torch.device("cpu"), restrict_impact=-1, var_group='glob'):
    if magn == 0:
        return sample

    seed = 0
    np.random.seed(seed)

    with torch.no_grad():
        if var_group == 'glob':
            noise = torch.Tensor(np.random.normal(offset,magn,(len(sample),vars_per_candidate[var_group]))).to(dev)
        else:
            noise = torch.Tensor(np.random.normal(offset,magn,(len(sample),cands_per_variable[var_group],vars_per_candidate[var_group]))).to(dev)
        xadv = sample + noise

        if var_group == 'glob':
            for i in range(vars_per_candidate['glob']):
                if i in integer_per_variable[var_group]:
                    xadv[:,i] = sample[:,i]
                else: # non integer, but might have defaults that should be excluded from shift
                    defaults = sample[:,i].cpu() == defaults_per_variable[var_group][i]
                    if torch.sum(defaults) != 0:
                        xadv[:,i][defaults] = sample[:,i][defaults]

                    if restrict_impact > 0:
                        difference = xadv[:,i] - sample[:,i]
                        allowed_perturbation = restrict_impact * torch.abs(sample[:,i])
                        high_impact = torch.abs(difference) > allowed_perturbation

                        if torch.sum(high_impact)!=0:
                            xadv[high_impact,i] = sample[high_impact,i] + allowed_perturbation[high_impact] * torch.sign(noise[high_impact,i])

        else:
            for j in range(cands_per_variable[var_group]):
                for i in range(vars_per_candidate[var_group]):
                    if i in integer_variables_by_candidate[var_group]:
                        xadv[:,j,i] = sample[:,j,i]
                    else:
                        defaults = sample[:,j,i].cpu() == defaults_per_variable[var_group][i]
                        if torch.sum(defaults) != 0:
                            xadv[:,j,i][defaults] = sample[:,j,i][defaults]

                        if restrict_impact > 0:
                            difference = xadv[:,j,i] - sample[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(sample[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv[high_impact,j,i] = sample[high_impact,j,i] + allowed_perturbation[high_impact] * torch.sign(noise[high_impact,j,i])       

        return xadv

def first_order_attack(epsilon=1e-2,sample=None,targets=None,thismodel=None,thiscriterion=None,reduced=True, dev=torch.device("cpu"), restrict_impact=-1, epsilon_factors=None, defaults_per_variable=None, do_sign_or_normed_grad = 'FGSM'):
    if epsilon == 0:
        return sample

    #glob, cpf, npf, vtx, cpf_pts, npf_pts, vtx_pts = sample
    cpf, npf, vtx, cpf_pts, npf_pts, vtx_pts = sample
    #xadv_glob = glob.clone().detach()
    xadv_cpf = cpf.clone().detach()
    xadv_npf = npf.clone().detach()
    xadv_vtx = vtx.clone().detach()
    xadv_cpf_pts = cpf_pts.clone().detach()
    xadv_npf_pts = npf_pts.clone().detach()
    xadv_vtx_pts = vtx_pts.clone().detach()

    #xadv_glob.requires_grad = True
    xadv_cpf.requires_grad = True
    xadv_npf.requires_grad = True
    xadv_vtx.requires_grad = True
    xadv_cpf_pts.requires_grad = True
    xadv_npf_pts.requires_grad = True
    xadv_vtx_pts.requires_grad = True

    #new_inpts = (xadv_glob,
    #             xadv_cpf,xadv_npf,xadv_vtx,
    #             xadv_cpf_pts,xadv_npf_pts,xadv_vtx_pts)
    new_inpts = (xadv_cpf,xadv_npf,xadv_vtx,
                 xadv_cpf_pts,xadv_npf_pts,xadv_vtx_pts)
    preds = thismodel(new_inpts)

    loss = thiscriterion(preds, targets).mean()

    thismodel.zero_grad(set_to_none=True)
    #loss.backward(retain_variables=True)
    loss.backward()

    with torch.no_grad():
        if do_sign_or_normed_grad == 'FGSM':
            #dx_glob = torch.sign(xadv_glob.grad.detach())
            dx_cpf = torch.sign(xadv_cpf.grad.detach())
            dx_npf = torch.sign(xadv_npf.grad.detach())
            dx_vtx = torch.sign(xadv_vtx.grad.detach())
            dx_cpf_pts = torch.sign(xadv_cpf_pts.grad.detach())
            dx_npf_pts = torch.sign(xadv_npf_pts.grad.detach())
            dx_vtx_pts = torch.sign(xadv_vtx_pts.grad.detach())
            #print(dx_cpf_pts.size())
        
        elif do_sign_or_normed_grad == 'NGM':
            #dx_cpf = torch.sign(xadv_cpf.grad.detach())
            #dx_npf = torch.sign(xadv_npf.grad.detach())
            #dx_vtx = torch.sign(xadv_vtx.grad.detach())
            #dx_cpf_pts = torch.sign(xadv_cpf_pts.grad.detach())
            #dx_npf_pts = torch.sign(xadv_npf_pts.grad.detach())
            #dx_vtx_pts = torch.sign(xadv_vtx_pts.grad.detach())
            dx_cpf = normalize_by_pnorm(xadv_cpf.grad.detach())
            dx_npf = normalize_by_pnorm(xadv_npf.grad.detach())
            dx_vtx = normalize_by_pnorm(xadv_vtx.grad.detach())
            dx_cpf_pts = torch.nan_to_num(normalize_by_pnorm(xadv_cpf_pts.grad.detach()))
            dx_npf_pts = torch.nan_to_num(normalize_by_pnorm(xadv_npf_pts.grad.detach()))
            dx_vtx_pts = torch.nan_to_num(normalize_by_pnorm(xadv_vtx_pts.grad.detach()))
            
        #print(dx_cpf, dx_cpf.size())
        #print(dx_npf, dx_npf.size())
        #print(dx_vtx, dx_vtx.size())
        #print(dx_cpf_pts, dx_cpf_pts.size())
        #print(dx_npf)
        #print(dx_vtx)
        #print(dx_cpf_pts)
        #print(dx_npf_pts)
        #print(dx_vtx_pts)

        #xadv_glob += epsilon * epsilon_factors['glob'] * dx_glob
        xadv_cpf += epsilon * epsilon_factors['cpf'] * dx_cpf
        xadv_npf += epsilon * epsilon_factors['npf'] * dx_npf
        xadv_vtx += epsilon * epsilon_factors['vtx'] * dx_vtx
        xadv_cpf_pts += epsilon * epsilon_factors['cpf_pts'] * dx_cpf_pts
        xadv_npf_pts += epsilon * epsilon_factors['npf_pts'] * dx_npf_pts
        xadv_vtx_pts += epsilon * epsilon_factors['vtx_pts'] * dx_vtx_pts

        if reduced:
            #for i in range(vars_per_candidate['glob']):
            #    if i in integer_variables_by_candidate['glob']:
            #        xadv_glob[:,i] = glob[:,i]
            #    else: # non integer, but might have defaults that should be excluded from shift
            #        defaults_glob = glob[:,i].cpu() == defaults_per_variable['glob'][i]
            #        if torch.sum(defaults_glob) != 0:
            #            xadv_glob[:,i][defaults_glob] = glob[:,i][defaults_glob]

            #        if restrict_impact > 0:
            #            difference = xadv_glob[:,i] - glob[:,i]
            #            allowed_perturbation = restrict_impact * torch.abs(glob[:,i])
            #            high_impact = torch.abs(difference) > allowed_perturbation

            #            if torch.sum(high_impact)!=0:
            #                xadv_glob[high_impact,i] = glob[high_impact,i] + allowed_perturbation[high_impact] * dx_glob[high_impact,i]

            for j in range(cands_per_variable['cpf']):
                for i in range(vars_per_candidate['cpf']):
                    if i in integer_variables_by_candidate['cpf']:
                        xadv_cpf[:,j,i] = cpf[:,j,i]
                    else:
                        #defaults_cpf = cpf[:,j,i] == defaults_per_variable['cpf'][i]
                        defaults_cpf = torch.eq(cpf[:,j,i], defaults_per_variable['cpf'][i])
                        #if torch.sum(defaults_cpf) != 0:
                        #    xadv_cpf[:,j,i][defaults_cpf] = cpf[:,j,i][defaults_cpf]
                            
                        xadv_cpf[:,j,i] = (defaults_cpf) * cpf[:,j,i] + (~defaults_cpf) * xadv_cpf[:,j,i]

                        if restrict_impact > 0:
                            difference = xadv_cpf[:,j,i] - cpf[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(cpf[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv_cpf[high_impact,j,i] = cpf[high_impact,j,i] + allowed_perturbation[high_impact] * dx_cpf[high_impact,j,i]        

            for j in range(cands_per_variable['npf']):
                for i in range(vars_per_candidate['npf']):
                    if i in integer_variables_by_candidate['npf']:
                        xadv_npf[:,j,i] = npf[:,j,i]
                    else:
                        defaults_npf = npf[:,j,i] == defaults_per_variable['npf'][i]
                        #if torch.sum(defaults_npf) != 0:
                        #    xadv_npf[:,j,i][defaults_npf] = npf[:,j,i][defaults_npf]
                            
                        xadv_npf[:,j,i] = (defaults_npf) * npf[:,j,i] + (~defaults_npf) * xadv_npf[:,j,i]

                        if restrict_impact > 0:
                            difference = xadv_npf[:,j,i] - npf[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(npf[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv_npf[high_impact,j,i] = npf[high_impact,j,i] + allowed_perturbation[high_impact] * dx_npf[high_impact,j,i]   

            for j in range(cands_per_variable['vtx']):
                for i in range(vars_per_candidate['vtx']):
                    if i in integer_variables_by_candidate['vtx']:
                        xadv_vtx[:,j,i] = vtx[:,j,i]
                    else:
                        defaults_vtx = vtx[:,j,i] == defaults_per_variable['vtx'][i]
                        #if torch.sum(defaults_vtx) != 0:
                        #    xadv_vtx[:,j,i][defaults_vtx] = vtx[:,j,i][defaults_vtx]
                            
                        xadv_vtx[:,j,i] = (defaults_vtx) * vtx[:,j,i] + (~defaults_vtx) * xadv_vtx[:,j,i]

                        if restrict_impact > 0:
                            difference = xadv_vtx[:,j,i] - vtx[:,j,i]
                            allowed_perturbation = restrict_impact * torch.abs(vtx[:,j,i])
                            high_impact = torch.abs(difference) > allowed_perturbation

                            if torch.sum(high_impact)!=0:
                                xadv_vtx[high_impact,j,i] = vtx[high_impact,j,i] + allowed_perturbation[high_impact] * dx_vtx[high_impact,j,i] 
### NEW

            for j in range(cands_per_variable['cpf_pts']):
                for i in range(vars_per_candidate['cpf_pts']):
                    #if i in integer_variables_by_candidate['cpf_pts']:
                    #    xadv_cpf_pts[:,j,i] = cpf_pts[:,j,i]
                    #else:
                    defaults_cpf_pts = cpf_pts[:,j,i] == defaults_per_variable['cpf_pts'][i]
                    #if torch.sum(defaults_cpf_pts) != 0:
                    #    xadv_cpf_pts[:,j,i][defaults_cpf_pts] = cpf_pts[:,j,i][defaults_cpf_pts]
                            
                    xadv_cpf_pts[:,j,i] = (defaults_cpf_pts) * cpf_pts[:,j,i] + (~defaults_cpf_pts) * xadv_cpf_pts[:,j,i]

                    if restrict_impact > 0:
                        difference = xadv_cpf_pts[:,j,i] - cpf_pts[:,j,i]
                        allowed_perturbation = restrict_impact * torch.abs(cpf_pts[:,j,i])
                        high_impact = torch.abs(difference) > allowed_perturbation

                        if torch.sum(high_impact)!=0:
                            xadv_cpf_pts[high_impact,j,i] = cpf_pts[high_impact,j,i] + allowed_perturbation[high_impact] * dx_cpf_pts[high_impact,j,i]

            for j in range(cands_per_variable['npf_pts']):
                for i in range(vars_per_candidate['npf_pts']):
                    #if i in integer_variables_by_candidate['npf_pts']:
                    #    xadv_npf_pts[:,j,i] = npf_pts[:,j,i]
                    #else:
                    defaults_npf_pts = npf_pts[:,j,i] == defaults_per_variable['npf_pts'][i]
                    #if torch.sum(defaults_npf_pts) != 0:
                    #    xadv_npf_pts[:,j,i][defaults_npf_pts] = npf_pts[:,j,i][defaults_npf_pts]
                    
                    xadv_npf_pts[:,j,i] = (defaults_npf_pts) * npf_pts[:,j,i] + (~defaults_npf_pts) * xadv_npf_pts[:,j,i]

                    if restrict_impact > 0:
                        difference = xadv_npf_pts[:,j,i] - npf_pts[:,j,i]
                        allowed_perturbation = restrict_impact * torch.abs(npf_pts[:,j,i])
                        high_impact = torch.abs(difference) > allowed_perturbation

                        if torch.sum(high_impact)!=0:
                            xadv_npf_pts[high_impact,j,i] = npf_pts[high_impact,j,i] + allowed_perturbation[high_impact] * dx_npf_pts[high_impact,j,i]

            for j in range(cands_per_variable['vtx_pts']):
                for i in range(vars_per_candidate['vtx_pts']):
                    #if i in integer_variables_by_candidate['vtx_pts']:
                    #    xadv_vtx_pts[:,j,i] = vtx_pts[:,j,i]
                    #else:
                    defaults_vtx_pts = vtx_pts[:,j,i] == defaults_per_variable['vtx_pts'][i]
                    #if torch.sum(defaults_vtx_pts) != 0:
                    #    xadv_vtx_pts[:,j,i][defaults_vtx_pts] = vtx_pts[:,j,i][defaults_vtx_pts]
                    
                    xadv_vtx_pts[:,j,i] = (defaults_vtx_pts) * vtx_pts[:,j,i] + (~defaults_vtx_pts) * xadv_vtx_pts[:,j,i]
                    
                    if restrict_impact > 0:
                        difference = xadv_vtx_pts[:,j,i] - vtx_pts[:,j,i]
                        allowed_perturbation = restrict_impact * torch.abs(vtx_pts[:,j,i])
                        high_impact = torch.abs(difference) > allowed_perturbation

                        if torch.sum(high_impact)!=0:
                            xadv_vtx_pts[high_impact,j,i] = vtx_pts[high_impact,j,i] + allowed_perturbation[high_impact] * dx_vtx_pts[high_impact,j,i]
                                
#        return xadv_glob.detach(),xadv_cpf.detach(),xadv_npf.detach(),xadv_vtx.detach(),xadv_cpf_pts.detach(),xadv_npf_pts.detach(),xadv_vtx_pts.detach()
        return xadv_cpf.detach(),xadv_npf.detach(),xadv_vtx.detach(),xadv_cpf_pts.detach(),xadv_npf_pts.detach(),xadv_vtx_pts.detach()
