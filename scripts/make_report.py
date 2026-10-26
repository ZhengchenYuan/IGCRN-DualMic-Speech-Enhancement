"""Generate exactly one A4 page; all prose is Simplified Chinese."""
from pathlib import Path
import pymupdf as fitz

path=Path('report/report_zh.pdf'); path.parent.mkdir(exist_ok=True)
doc=fitz.open(); page=doc.new_page(width=595.2756,height=841.8898)
page.draw_rect(fitz.Rect(0,0,595.2756,83),color=None,fill=(.10,.18,.28))
page.insert_text((40,42),'双麦克风语音增强：聚焦复现报告',fontname='china-s',fontsize=21,color=(1,1,1))
page.insert_text((40,64),'实现与实验模板　｜　研究训练及评测尚未完成　｜　日期：二〇二六年十月二十六日',fontname='china-s',fontsize=9,color=(1,1,1))
y=104

def block(title,body):
    global y
    page.insert_text((40,y),title,fontname='china-s',fontsize=12,color=(.08,.34,.43)); y+=10
    body=body.replace('<br/>','\n')
    # Explicit CJK character wrapping makes the one-page geometry auditable.
    lines=[]
    for paragraph in body.split('\n'):
        line=''; width=0
        for ch in paragraph:
            cw=10 if ord(ch)>255 else 5.5
            if width+cw>515:
                lines.append(line); line=''; width=0
            line+=ch; width+=cw
        if line: lines.append(line)
    for line in lines:
        y+=14
        page.insert_text((40,y),line,fontname='china-s',fontsize=10,color=(.14,.20,.28))
    y+=22

block('一、架构与复现范围','依据刘金江、张雪良于二〇二一年发表的原位门控卷积循环网络论文实现。双通道复数谱拆为四个实数通道；六层五乘一、步长为一的门控卷积保留频率分辨率。各频点复用同一两层双向长短期记忆网络；幅度和相位由独立跳接解码器预测。幅度采用掩蔽加映射，相位归一化，再经逆短时傅里叶变换还原波形。保留二百五十七个频点，解决原文频点及输入通道描述歧义。')
block('二、数据与声学模拟','使用依法公开的英语有声读物语音库及公开音乐语音噪声库中的噪声，保留署名与许可信息；按说话人、噪声录音划分互斥集合。采样率一万六千赫兹，房间五乘五乘三米，麦距二厘米。固定种子二〇二六，采用直达声及六个一阶墙面镜像生成脉冲响应；分别卷积后按第一麦克风能量配置负三、零、正三分贝信噪比。目标为第一麦克风含房间响应的无噪语音。保存源文件哈希、位置、截取偏移、增益及评测清单。')
block('三、训练与公平消融','计划使用六十四通道完整模型，三十二毫秒窗、十六毫秒帧移、平方根汉宁窗；幅度及复数谱立方根压缩均方损失。自适应矩估计优化器学习率零点〇〇〇二，批量一，三十轮，两秒片段，梯度范数裁剪五。双麦克风与第一麦克风重复输入两组从头独立训练，采用相同初始化种子、划分、架构和预算。谷歌云端笔记本使用指定显卡，云盘持久保存权重、优化器及随机状态；显存占用仍待实测。')
block('四、客观结果与听感案例','以含混响目标计算尺度不变信号失真比提升量与短时客观可懂度，逐句平均并按信噪比分组；均不作额外时移对齐。<br/>未处理第一麦克风：提升量为零（定义），可懂度待测。<br/>第一加第二麦克风：提升量待测，可懂度待测。<br/>独立训练的第一加第一麦克风：提升量待测，可懂度待测。<br/>成功案例待填：记录编号、信噪比、指标及噪声抑制听感。失败案例待填：重点检查近角度声源、弱语音与房间失配造成的残噪或失真。现有四份合成诊断音频仅检查管线；真实语音增强案例需训练后补充，不构成效果证据。')
block('五、局限与人工智能辅助披露','此为核心结构复现，替代语料、一阶镜像响应、非负幅度激活及频点处理均偏离原文；不复现论文数据规模，也不比较其原始失真比分数。双向网络需离线处理，尚无泛化或实时性结论。人工智能辅助编写代码、测试、文档和本报告；应由提交者审核原理、运行实验并试听。所有待测字段仅在真实训练与评测完成后填写，不虚构数值或成功案例。')
assert y<790, y
page.draw_line((40,803),(555,803),color=(.75,.8,.83))
page.insert_text((40,819),'参考：刘、张，《原位门控卷积循环神经网络用于双通道语音增强》，国际语音通信会议，二〇二一年。',fontname='china-s',fontsize=8,color=(.35,.4,.45))
doc.set_metadata({'title':'双麦克风语音增强复现报告','author':'项目提交者'})
doc.save(path)
print(path)
