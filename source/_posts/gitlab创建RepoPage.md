---
layout: 工具介绍
title: gitlab创建RepoPage
date: 2020-04-09 22:44:50
category:
- 工具介绍
---

## 步骤 1：创建项目
打开 gitlab，点击上方的“+”号按钮，在显示的页面中填入 repo 相关的信息，例如 Project name 和 description。最后，点击 Create project 按钮。

![步骤 1](https://raw.githubusercontent.com/clvsit/markdown-image/master/eigen/tools/20200409231825.jpg)

## 步骤 2：添加 License
在斌哥提供的 Youtubu 教程中需要创建 License，但我在实践的过程中发现该步骤非必需。

创建项目后会自动跳转到 repo 的 detail 页面，在该页面的正中央有一排按钮组，点击“Add License”。

![步骤 2-1](https://raw.githubusercontent.com/clvsit/markdown-image/master/eigen/tools/20200409232146.jpg)

然后，在调整后的页面中，选择 Template -> Apache License 2.0。

![步骤 2-1](https://raw.githubusercontent.com/clvsit/markdown-image/master/eigen/tools/20200409232614.jpg)

## 步骤 3：添加 .gitlab-ci 配置文件
完成 License 添加后，回退到 repo detail 页面，此时已有 LICENSE 文件。

![步骤 3-1](https://raw.githubusercontent.com/clvsit/markdown-image/master/eigen/tools/20200409232824.jpg)

然后按图所示，创建一个新的文件。

![步骤 3-2](https://raw.githubusercontent.com/clvsit/markdown-image/master/eigen/tools/20200409232928.jpg)

在创建新文件的页面中，先在左侧的 Template 下拉框中选择 **.gitlab-ci.yml**，然后在右侧的下拉框中选择 **HTML**。最后，点击 Commit changes。

此时，我们查看左侧栏的 CI/CD -> Pipelines，可以看到已经起了一个 Pipelines。

![步骤 3-3](https://raw.githubusercontent.com/clvsit/markdown-image/master/eigen/tools/20200409233216.jpg)

## 步骤 4：创建 or 上传静态网页
我们继续创建 index.html 文件。

![步骤 4-1](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/gitlab%20page7.jpg)

创建完成后，点击左侧栏的 Pages，此时在页面中央的 Access pages 中出现了链接，点击该链接即可看到我们刚才创建的 index.html 内容。

![步骤 4-2](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/gitlab%20page8.jpg)