# Example HW settings

## General introduction
In this repository, we have modeled 5 well-known DNN accelerators, which are Ascend [1], Edge TPU [2],
Meta prototype [3], Tesla NPU [4], and TPU [5], for our depth-first scheduling research.
To make a fair and relevant comparison, we normalized all of them to have 1024 MACs and maximally 2MB global buffer (GB) 
but kept their spatial unrolling and local buffer settings, as shown in Table I Idx 1/3/5/7/9.
Besides, we constructed a variant of every normalized architecture (by changing its on-chip memory hierarchy), denoted with ‘DF’ in the
end of the name, as shown in Table I Idx 2/4/6/8/10.

## Specific settings

Table I:
![image](https://user-images.githubusercontent.com/55059827/183848886-c85b9950-5e49-47c9-8a47-ad05062debc3.png)
Neural network layer dimension notation: 

K is for output channel; C is for input channel; OX and OY are feature map spatial dimensions; FX and FY are weight spatial dimensions.



---
Reference

[1] H. Liao, J. Tu, J. Xia, H. Liu, X. Zhou, H. Yuan, and Y. Hu,
“Ascend: a scalable and unified architecture for ubiquitous deep neural
network computing : Industry track paper,” in 2021 IEEE International
Symposium on High-Performance Computer Architecture (HPCA), 2021,
pp. 789–801.

[2] C.-T. Huang, Y.-C. Ding, H.-C. Wang, C.-W. Weng, K.-P. Lin, L.-W.
Wang, and L.-D. Chen, “Ecnn: A block-based and highly-parallel cnn
accelerator for edge inference,” in Proceedings of the 52nd Annual
IEEE/ACM International Symposium on Microarchitecture, ser. MICRO
’52. New York, NY, USA: Association for Computing Machinery,
2019, p. 182–195.

[3] H. E. Sumbul, T. F. Wu, Y. Li, S. S. Sarwar, W. Koven, E. Murphy-
Trotzky, X. Cai, E. Ansari, D. H. Morris, H. Liu, D. Kim, E. Beigne,
R. Labs, and Meta, “System-level design and integration of a prototype
ar/vr hardware featuring a custom low-power dnn accelerator chip in
7nm technology for codec avatars,” in 2022 IEEE Custom Integrated
Circuits Conference (CICC), 2022, pp. 01–08.

[4] E. Talpes, D. D. Sarma, G. Venkataramanan, P. Bannon, B. McGee,
B. Floering, A. Jalote, C. Hsiong, S. Arora, A. Gorti, and G. S. Sachdev,
“Compute solution for tesla’s full self-driving computer,” IEEE Micro,
vol. 40, no. 2, pp. 25–35, 2020.

[5] N. P. Jouppi, C. Young, N. Patil, D. Patterson, G. Agrawal, R. Bajwa,
S. Bates, S. Bhatia, N. Boden, A. Borchers, R. Boyle, P.-l. Cantin,
C. Chao, C. Clark, J. Coriell, M. Daley, M. Dau, J. Dean, B. Gelb, T. V.
Ghaemmaghami, R. Gottipati, W. Gulland, R. Hagmann, C. R. Ho,
D. Hogberg, J. Hu, R. Hundt, D. Hurt, J. Ibarz, A. Jaffey, A. Jaworski,
A. Kaplan, H. Khaitan, D. Killebrew, A. Koch, N. Kumar, S. Lacy,
J. Laudon, J. Law, D. Le, C. Leary, Z. Liu, K. Lucke, A. Lundin,
G. MacKean, A. Maggiore, M. Mahony, K. Miller, R. Nagarajan,
R. Narayanaswami, R. Ni, K. Nix, T. Norrie, M. Omernick,
N. Penukonda, A. Phelps, J. Ross, M. Ross, A. Salek, E. Samadiani,
C. Severn, G. Sizikov, M. Snelham, J. Souter, D. Steinberg, A. Swing,
M. Tan, G. Thorson, B. Tian, H. Toma, E. Tuttle, V. Vasudevan,
R. Walter, W. Wang, E. Wilcox, and D. H. Yoon, “In-datacenter
performance analysis of a tensor processing unit,” SIGARCH Comput.
Archit. News, vol. 45, no. 2, p. 1–12, jun 2017. 
