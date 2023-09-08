// src/App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Header from './components/Header';
import BlogPosts from './components/BlogPosts';
import BlogPostPage from './components/BlogPostPage';
import About from './components/About';
// import CV from './components/CV';
// import Other from './components/Other';


function App() {
  return (
    <Router>
      <div className="App bg-gray-900 text-gray-100 min-h-screen">
        <Header />
        <Routes>
          <Route path="/" element={<BlogPosts />} />
          <Route path="/blogs/:url" element={<BlogPostPage />} />
          <Route path="/about" element={<About />} />
          {/* <Route path="/cv" element={<CV />} /> */}
          {/* <Route path="/other" element={<Other />} /> */}
          {/* Other routes can be added here */}
        </Routes>
      </div>
    </Router>
  );
}

export default App;
