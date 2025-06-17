// src/components/CV.js
import React from 'react';
import ReactMarkdown from 'react-markdown';


const markdownContent = `
# **Curriculum Vitae**

## **Profile**

A research engineer passionate about the intersection of robotics, AI, and reinforcement learning. I believe these fields are critical to solving complex, real-world problems. I am seeking a role where I can further advance these technologies and collaborate on research-driven engineering projects.

I write extensively about my personal projects in these areas on my [blog](www.mauicv.com). Most recently, I trained a robot dog to walk. To do this, I recreated the [DayDreamer](https://arxiv.org/abs/2206.14176) paper by Wu et al., but used a transformer as the world model instead of an RNN. For more details, please see the [blog post](https://mauicv.com/#/posts/real-world-model-rl).

A project that's a personal favourite of mine is my basic JavaScript [physics engine](https://genesistubs.com/asteroids) that I built when I first got into software development.

---

## **Education**

**Doctorate in Mathematics**  
*Imperial College London* — October 2013 to October 2018  
Thesis title: *Bifurcations of Set-Valued Dynamical Systems*

**MMath (Hons) in Mathematics**  
*University of Warwick* — October 2008 to October 2012

---

## **Experience**

### **Research Engineer**  
*Seldon* — London, 2021 to 2024

- **LLM Module Development**: Led the design and implementation of a Python-based runtime for deploying large language models (LLMs) using [Seldon Core v2](https://docs.seldon.ai/seldon-core-2). This solution extended the capabilities of [MLServer](https://github.com/SeldonIO/MLServer), improving the serving infrastructure for generative AI models.
- **Open-Source Contributions**: Maintained and enhanced Seldon's open-source libraries, [Alibi-Explain](https://github.com/SeldonIO/alibi) and [Alibi-Detect](https://github.com/SeldonIO/alibi-detect), for explainability, drift, and outlier detection. Focused on integrating outlier detection methods and developing similarity-based methods for explainability.
- **Conference Speaker**: Delivered talks at key conferences such as ODSC Europe and [MLPrague](https://slideslive.com/39002425/machine-learning-explainability-understanding-model-decisions?ref=recommended) (800+ attendees), presenting on explainability using Alibi-Explain.

### **Software Engineer**  
*Allied Offsets (formerly Allied Crowds)* — London, 2018 to 2021

- **Carbon Offset Registry**: Solely responsible for building and deploying the company's carbon offsets registry, which tracks offsetting projects and credits. This product generated over £1 million in Annual Recurring Revenue (ARR).
- **Full Stack Development**: Developed the platform as a full-stack engineer, utilizing Python, JavaScript, and various web technologies to create a scalable, robust system.

---

## **Technical Skills**

### **Machine Learning & AI**

- **Reinforcement Learning**: Extensive hands-on experience implementing state-of-the-art reinforcement learning (RL) models. Notable projects include:
    - [Trained a robot to walk](https://mauicv.com/#/posts/real-world-model-rl): Achieved this using only real-world experience and a learned world model based on a transformer architecture.
    - [World Model RL](https://github.com/mauicv/world-model-rl): Implemented RL agents using RNN- and Transformer-based world models, inspired by the [DreamerV1](https://arxiv.org/abs/1912.01603), [DreamerV2](https://arxiv.org/abs/2010.02193), and [TSSM](https://arxiv.org/html/2202.09481v2) papers.
    - [Transformers Library](https://github.com/mauicv/transformers): Developed a custom transformers library implementing features such as [Mixture of Experts](https://mauicv.com/#/posts/moe-expert-choice) and relative positional encoding.
- **Generative AI**: Expertise in generative models, particularly VQ-VAE and discrete diffusion models, leveraging transformers for fine-grained image generation. Published detailed blog posts on these methods ([1](https://mauicv.com/#/posts/perceptual-loss-for-vaes), [2](https://mauicv.com/#/posts/vqvaes-and-perceptual-losses), and [3](https://mauicv.com/#/posts/generative-modelling-using-vq-vaes)).

## **Programming Languages & Frameworks**

- **Python** (8 years): Proficient in using Python for machine learning, data analysis, and web development. Expertise with libraries such as **PyTorch**, **TensorFlow**, **NumPy**, **Pandas**, and **FastAPI**.
- **JavaScript** (4 years): Experience with modern web development frameworks, including **React** and **VueJS**. Developed a physics engine ([Genesis Tubs](https://github.com/mauicv/genesis-tubs-engine)) for [browser-based games](https://genesistubs.com/asteroids).
- **Kubernetes**: Extensive experience with container orchestration and deployment, particularly using [Seldon Core](https://docs.seldon.ai/seldon-core-2) to deploy machine learning models on **GKE** clusters with GPU support.
- **SQL/PostgreSQL**: Solid understanding of SQL for data management and querying in production systems.
- **C/C++**: I have completed courses in C/C++ and have used it for some projects.

---

### **Interests**

Outside of work, I love to boulder and regularly project at V8/9 grades. I also enjoy running and recently achieved a sub-18-minute 5K. In my downtime, I enjoy reading and playing guitar. My greatest joy in life is good food with good friends.

---

### **References**

Available upon request.

---

### **Links**

- GitHub: [github.com/mauicv](https://github.com/mauicv)
- Blog: [mauicv.com](https://mauicv.com/)
- LinkedIn: [linkedin.com/in/alex-athorne/](https://www.linkedin.com/in/alex-athorne/)

`

const CV = () => {
  return (
    <div className="flex flex-col md:flex-row items-center justify-center mt-10 w-2/3 mx-auto">
      {/* Text */}
      <div className="w-full md:w-full p-4">
        <p className="text-base md:text-lg">
          <ReactMarkdown 
            className="markdown-content"
            children={markdownContent}
          />
        </p>
      </div>
    </div>
  );
}

export default CV;
