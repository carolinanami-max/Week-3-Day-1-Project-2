import os
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from llm_integration import LLMIntegration
from prompt_templates import PromptTemplates

# Load environment variables
load_dotenv()

class PersonalBrandContentCreator:
    def __init__(self):
        print("🚀 Initializing Personal Brand Content Creator...")
        
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.llm = LLMIntegration(api_key=os.getenv("OPENAI_API_KEY"))
        self.templates = PromptTemplates()
        
        # Load knowledge bases
        self.load_knowledge_bases()
    
    def load_knowledge_bases(self):
        """Load your personal brand documents"""
        print("\n📚 Loading your personal brand knowledge...")
        self.doc_processor.load_all()
        print("✅ Ready to create content!\n")
    
    def create_linkedin_post(self, topic: str, expertise: str = "", audience: str = "my network"):
        """Generate an authentic LinkedIn post"""
        print(f"🔍 Finding relevant context for: '{topic}'")
        
        # Get relevant context from your knowledge bases
        context = self.doc_processor.search(topic)
        
        # Get template and format
        template = self.templates.linkedin_post()
        prompt = template.format(
            context=context,
            topic=topic,
            expertise=expertise,
            audience=audience
        )
        
        # Generate content
        print("✍️ Writing your LinkedIn post...")
        content = self.llm.generate(prompt)
        
        return content
    
    def create_carousel(self, topic: str, perspective: str = ""):
        """Generate a LinkedIn carousel post"""
        context = self.doc_processor.search(topic)
        template = self.templates.linkedin_carousel()
        prompt = template.format(
            context=context,
            topic=topic,
            perspective=perspective
        )
        return self.llm.generate(prompt)
    
    def create_thought_leadership(self, topic: str, angle: str = ""):
        """Generate a thought leadership post"""
        context = self.doc_processor.search(topic)
        template = self.templates.thought_leadership()
        prompt = template.format(
            context=context,
            topic=topic,
            angle=angle
        )
        return self.llm.generate(prompt)

def main():
    # Create the content creator
    creator = PersonalBrandContentCreator()
    
    # Interactive menu
    while True:
        print("\n" + "="*50)
        print("PERSONAL BRAND LINKEDIN CONTENT CREATOR")
        print("="*50)
        print("1. Create authentic LinkedIn post")
        print("2. Create LinkedIn carousel")
        print("3. Create thought leadership post")
        print("4. Exit")
        
        choice = input("\nChoose an option (1-4): ")
        
        if choice == "4":
            print("👋 Good luck with your personal brand!")
            break
        
        topic = input("What topic do you want to post about? ")
        
        if choice == "1":
            expertise = input("Your expertise in this area (optional): ")
            content = creator.create_linkedin_post(topic, expertise)
            print("\n" + "="*50)
            print("YOUR LINKEDIN POST:")
            print("="*50)
            print(content)
            
        elif choice == "2":
            perspective = input("Your unique perspective (optional): ")
            content = creator.create_carousel(topic, perspective)
            print("\n" + "="*50)
            print("YOUR CAROUSEL POST:")
            print("="*50)
            print(content)
            
        elif choice == "3":
            angle = input("Your unique angle (optional): ")
            content = creator.create_thought_leadership(topic, angle)
            print("\n" + "="*50)
            print("YOUR THOUGHT LEADERSHIP POST:")
            print("="*50)
            print(content)
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()