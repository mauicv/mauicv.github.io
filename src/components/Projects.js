// src/components/BlogPosts.js
import React from 'react';
import { Link } from 'react-router-dom';


function Projects() {
  return (
    <div className="container mx-auto px-4 py-12 md:w-1/2">
      <h1 className="text-3xl font-bold mb-6">Projects</h1>
      <Link to='https://genesistubs.com/asteroids' className="block">
        <div className="bg-gray-800 p-4 my-4 rounded-lg flex items-center hover:bg-gray-700 transition duration-200">
          <img 
            src='/images/genesis-tub-tooth.png' 
            alt="Link to genesis tubs games website" 
            className="w-1/4 rounded-lg mr-4" 
          />
          <div className="w-3/4">
            <h2 className="text-xl font-bold">www.genesistubs.com</h2>
            <p>
              I built a physics engine in javascript. This links to a website I built to show some
              of the games I made with it. I did it a long time ago, and I would do it differently now.
              But i'm still quite pleased with it. The code is available on my github.
            </p>
          </div>
        </div>
      </Link>

      <Link to='https://github.com/mauicv/gerel' className="block">
        <div className="bg-gray-800 p-4 my-4 rounded-lg flex items-center hover:bg-gray-700 transition duration-200">
          <img 
            src='/images/evo-ant-8.gif' 
            alt="Link to gerel repo"
            className="w-1/4 rounded-lg mr-4" 
          />
          <div className="w-3/4">
            <h2 className="text-xl font-bold">GeRel</h2>
            <p>
              gerel stands for genetic algorithms for reinforcement learning. It's a library I wrote in
              python to make it easy to use genetic algorithms to train neural networks to solve RL 
              problems. It should be a whole blog post and at some point it will be. But for now, check
              out the repo.
            </p>
          </div>
        </div>
      </Link>

      <Link to='https://github.com/mauicv/evo-quad' className="block">
        <div className="bg-gray-800 p-4 my-4 rounded-lg flex items-center hover:bg-gray-700 transition duration-200">
          <img 
            src='/images/quadruped-progress.png' 
            alt="Link to evo-quad repo" 
            className="w-1/4 rounded-lg mr-4" 
          />
          <div className="w-3/4">
            <h2 className="text-xl font-bold">Evo-Quad</h2>
            <p>
                Evo-Quad uses the gerel library and applies it to a simulation of a quadrupedal robot. The idea was to
                eventually use it to train a real robot. I did manage to train the simulation to walk, but I never got
                around to building the real robot. The code is available on my github but its something I stopped halfway
                so its more of a work site. This should also be a whole blog post...
            </p>
          </div>
        </div>
      </Link>

      <Link to='https://chrome.google.com/webstore/detail/build-a-bit/amkfanieebhiifhcnnflgnkbpeffmjlb' className="block">
        <div className="bg-gray-800 p-4 my-4 rounded-lg flex items-center hover:bg-gray-700 transition duration-200">
          <img 
            src='/images/build-a-bit.png' 
            alt="Link to build-a-bit chrome extension" 
            className="w-1/4 rounded-lg mr-4" 
          />
          <div className="w-3/4">
            <h2 className="text-xl font-bold">Build-a-bit</h2>
            <p>
              I built a chrome extension that helps you manage and create new habits. No one uses it, except me. 
            </p>
          </div>
        </div>
      </Link>

    </div>
  );}

export default Projects;
