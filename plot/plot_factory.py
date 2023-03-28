# Project:
#   VQA
# Description:
#   Plotting functions
# Author: 
#   Sergio Tascon-Morales

import numpy as np
import matplotlib.pyplot as plt
from os.path import join as jp
import pandas as pd



def plot_learning_curve(metric_dict_train, metric_dict_val, metric_name, x_label='epoch', title="Learning curve", save=False, path=None):
    """ Input dictionaries are expected to have epoch indexes (string) as keys and floats as values"""
    fig = plt.figure()
    if metric_name == 'loss':
        top_val = max(max(list(metric_dict_train.values())), max(list(metric_dict_val.values()))) 
    else:
        top_val = 1.0
        metric_name = metric_name.upper()
        
    # plot train metrics
    plt.plot([int(e) for e in metric_dict_train.keys()], list(metric_dict_train.values()), label=metric_name + ' train', linewidth=2, color='orange')
    # plot val metrics
    plt.plot([int(e) for e in metric_dict_val.keys()], list(metric_dict_val.values()), label=metric_name + ' val', linewidth=2, color='blue')
    plt.xticks([int(e) for e in metric_dict_train.keys()])
    plt.grid()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylim((0, top_val))
    plt.ylabel(metric_name)
    plt.legend()
    if save:
        if path is not None:
            plt.savefig(jp(path, metric_name + '.png'), dpi=300)
        else:
            raise ValueError


def plot_roc_prc(roc, auc, prc, ap, title='ROC and PRC plots', save=True, path=None, suffix=''):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    f.suptitle(title)
    # plot PRC
    ax1.plot(prc[1], prc[0], label = "PRC , AP: " + "{:.3f}".format(ap))
    #ax1.plot([0, 1], [no_skill, no_skill], linestyle='--', color = colors[k], label='No Skill')
    ax1.set_xlabel("recall")
    ax1.set_ylabel("precision")
    ax1.grid()
    ax1.legend()

    # plot ROC
    ax2.plot(roc[0], roc[1],label = "ROC, AUC: " + "{:.3f}".format(auc))
    #ax2.plot(fpr_dumb, tpr_dumb, linestyle="--", color = "gray", label="No Skill")
    ax2.set_xlabel("fpr")
    ax2.set_ylabel("tpr")
    ax2.grid()
    ax2.legend()

    if save and path is not None:
        plt.savefig(jp(path, 'ROC_PRC_' + suffix + '.png'), dpi=300)


def overlay_mask(img, mask, gt, save= False, path_without_ext=None, alpha = 0.7):
    masked = np.ma.masked_where(mask ==0, mask)
    gt = np.ma.masked_where(gt==0, gt)
    fig, ax = plt.subplots()
    ax.imshow(img, 'gray', interpolation='none')
    ax.imshow(masked, 'jet', interpolation='none', alpha=alpha)
    ax.imshow(gt, 'pink', interpolation='none', alpha=alpha)
    fig.set_facecolor("black")
    fig.tight_layout()
    ax.axis('off')
    if save:
        plt.savefig(path_without_ext + '.png', bbox_inches='tight')
    plt.show()

def plot_inconsistency_dme(img, mask_region, grade_gt, grade_pred, whole_gt, whole_pred, region_gt, region_pred, save= False, path_without_ext=None, alpha = 0.5):
    masked = np.ma.masked_where(mask_region == 0, mask_region)
    plt.ioff()    
    f = plt.figure()
    plt.imshow(img)
    plt.imshow(masked, 'jet', interpolation='none', alpha=alpha)
    plt.title('Grade GT: ' + str(grade_gt) + ', Grade Pred: ' + str(grade_pred) + '\n' + 'EX in image? GT: ' + str(whole_gt) + ', Pred: ' + str(whole_pred) + '\n' + 'EX in region? GT: ' + str(region_gt) + ', Pred: ' + str(region_pred))
    f.tight_layout()
    plt.axis('off')
    if save:
        plt.savefig(path_without_ext + '.png', bbox_inches='tight')


def overlay_windows_with_colors(img, windows, category, save= False, path_without_ext=None, alpha = 0.7):
    # shows image with colored windows. Different colors are used depending on TP, TN, FP, FN
    palette = {'TP': 'jet', 'TN': 'gray', 'FN': 'autumn', 'FP': 'Wistia'}
    fig, ax = plt.subplots()        
    ax.imshow(img, 'gray', interpolation='none')
    for cl, masks_curr_cl in windows.items():
        for mask in masks_curr_cl:
            masked = np.ma.masked_where(mask ==0, mask)
            ax.imshow(masked, palette[cl], interpolation='none', alpha=alpha)
    fig.set_facecolor("black")
    fig.tight_layout()
    title_obj = plt.title(category)
    plt.getp(title_obj)                    #print out the properties of title
    plt.getp(title_obj, 'text')            #print out the 'text' property for title
    plt.setp(title_obj, color='w')
    ax.axis('off')
    if save:
        plt.savefig(path_without_ext + '.png', bbox_inches='tight')
    else:
        plt.show()

def fcn2(x,y, gamma = 2):
    z = x*(2-y)
    z[z<0] = 0
    return z

def plot3d(x,y,z):
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
    ax.set_title('surface')
    plt.show()


def plot_performance_heatmaps(path_csv, file_names, cmap='bone', vmin=0, vmax=1):
    fig, axes = plt.subplots(1,len(file_names))
    for f,ax in zip(file_names, axes):
        df = pd.read_csv(jp(path_csv, f))   
        df_np = df.to_numpy()
        img = ax.imshow(df_np, cmap=cmap, vmin=vmin, vmax=vmax)
        x_lab = ['',0.5,1.0,1.5,2.0,2.5]
        y_lab = ['',0.1,0.2,0.3,0.4,0.5]
        ax.set_xticklabels(x_lab)
        ax.set_yticklabels(y_lab)
    fig.colorbar(img, ax=axes.ravel().tolist())
    plt.show()
    a = 42

def plot_boxplot_row(path_csv, prefix, lambda_digits, labels):
    data_row = []
    fig, axes = plt.subplots(1, len(lambda_digits), sharey=True)
    for di in lambda_digits:
        data_row.append(pd.read_csv(jp(path_csv, prefix + '_lambda0' + str(di) + '.csv')).to_numpy())
    for ax, row in zip(axes, data_row):
        ax.boxplot(row, labels=labels)
        ax.grid()
    plt.show()

if __name__ == '__main__':
    prefix = 'q3'
    path_csv = '/home/sergio814/Documents/PhD/docs/my_papers/MICCAI2022/boxplots'
    lambda_range = [1,2,3,4,5]
    gamma_labels = ['0.5','1.0','1.5', '2.0','2.5']
    plot_boxplot_row(path_csv, prefix, lambda_range, gamma_labels)
