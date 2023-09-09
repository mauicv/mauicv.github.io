// src/components/Header.js
import React from 'react';
import { Link } from 'react-router-dom';


function Header() {
  return (
    <header className="bg-gray-800 shadow-md">
      <div className="container mx-auto px-4 py-2 flex justify-between items-center">
        <div className="text-2xl font-bold text-gray-100">
          <Link to="/" className="text-gray-300 hover:text-gray-100">Mauicv</Link>
        </div>
        <nav>
          <ul className="flex space-x-4">
            <li><Link to="/" className="text-gray-300 hover:text-gray-100">Posts</Link></li>
            <li><Link to="/projects" className="text-gray-300 hover:text-gray-100">Projects</Link></li>
            <li><Link to="/about" className="text-gray-300 hover:text-gray-100">About</Link></li>
            {/* <li><Link to="/cv" className="text-gray-300 hover:text-gray-100">CV</Link></li> */}
          </ul>
        </nav>
      </div>
    </header>
  );
}

export default Header;
