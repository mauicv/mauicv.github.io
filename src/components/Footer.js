// Footer.js
function Footer() {
    return (
      <footer className="bg-gray-800 text-gray-300 mt-16 py-6">
        <div className="container mx-auto text-center">
          <h2 className="text-2xl font-semibold mb-4">mauicv's blog</h2>
  
          <div className="mb-4 space-x-4">
            <span>email: <a href="mailto:alexander.athorne@gmail.com" className="text-gray-400 hover:text-white transition duration-300">alexander.athorne@gmail.com</a></span>
            <span>github: <a href="https://github.com/mauicv" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-white transition duration-300">mauicv</a></span>
            <span>twitter: <a href="https://twitter.com/oblibob" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-white transition duration-300">@oblibob</a></span>
          </div>
  
          <p className="text-sm italic">
            Informal blog about things I'm interested in. Currently mostly Maths and Reinforcement learning. Stuff I've learnt, or failed to. I make no apologies for spelling.
          </p>
        </div>
      </footer>
    );
  }
  
  export default Footer;
  