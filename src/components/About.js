import React from 'react';
import ReactMarkdown from 'react-markdown';


const markdownContent = `
I'm a research engineer currently working at [seldon.io](https://www.seldon.io/) on [explainable AI](https://github.com/SeldonIO/alibi) and [drift](https://github.com/SeldonIO/alibi-detect). Prior to this I worked for a [start up](https://alliedoffsets.com/) that built databases of carbon-offsetting projects. Even prior to that, I obtained a PhD from Imperial College London, studying dynamical systems.

This blog is mostly about my faulty understanding of software, mathematics, data science and machine learning.

Feel free to email me at: alexander.athorne@gmail.com if you have any questions or discover some egregious error that can't go uncorrected. I'm also on [twitter](https://twitter.com/oblibob) but I don't use it except to post links, and i'm on [github](https://github.com/mauicv) if you want to see my code.

___

__Note__ : *I'm dyslexic and so some of my spelling and grammar is pretty bad. I had the choice of this either being an easy and quick process for creating and sharing content or a slow and unpleasant one that's going to put me off writing. I chose the former because I'd rather put things out than sit on them trying to make them perfect.*
`

const About = () => {
  return (
    <div className="flex flex-col md:flex-row items-center justify-center mt-10 w-2/3 mx-auto">
      {/* Image */}
      <div className="w-full md:w-1/3 p-4">
        <img src="/me.jpeg" alt="Profile" className="mx-auto rounded w-2/3 md:w-full" />
      </div>

      {/* Text */}
      <div className="w-full md:w-2/3 p-4">
        <h1 className="text-xl md:text-2xl font-bold mb-4">Hello, I'm Alex Athorne</h1>
        <p className="text-base md:text-lg">
          <ReactMarkdown 
            className="markdown-content"
            children={markdownContent}
          />
        </p>
      </div>
    </div>
  );
};

export default About;

