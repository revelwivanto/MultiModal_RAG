import unittest
import torch
import os
from unittest.mock import Mock, patch
from PIL import Image
import io
import base64
import pytest
from PIL import UnidentifiedImageError

from rag_code import (
    EmbedData,
    QdrantVDB_QB,
    Retriever,
    RAG,
    batch_iterate,
    image_to_base64
)

class TestUtils(unittest.TestCase):
    """Menguji fungsi utilitas"""
    
    def test_batch_iterate(self):
        """Menguji fungsi batch_iterate"""
        test_list = [1, 2, 3, 4, 5]
        batch_size = 2
        batches = list(batch_iterate(test_list, batch_size))
        self.assertEqual(batches, [[1, 2], [3, 4], [5]])

    def test_batch_iterate_empty_list(self):
        """Menguji batch_iterate dengan daftar kosong"""
        test_list = []
        batch_size = 2
        batches = list(batch_iterate(test_list, batch_size))
        self.assertEqual(batches, [])

    def test_batch_iterate_batch_size_larger_than_list(self):
        """Menguji batch_iterate ketika ukuran batch lebih besar dari daftar"""
        test_list = [1, 2, 3]
        batch_size = 5
        batches = list(batch_iterate(test_list, batch_size))
        self.assertEqual(batches, [[1, 2, 3]])

    def test_image_to_base64(self):
        """Menguji fungsi image_to_base64"""
        # Membuat gambar uji sederhana
        test_image = Image.new('RGB', (100, 100), color='red')
        base64_string = image_to_base64(test_image)
        
        # Memverifikasi bahwa output adalah string base64 yang valid
        self.assertTrue(isinstance(base64_string, str))
        # Memverifikasi bahwa dapat didekode kembali ke byte
        try:
            base64.b64decode(base64_string)
        except Exception as e:
            self.fail(f"Gagal mendekode string base64: {e}")

    def test_image_to_base64_different_formats(self):
        """Menguji image_to_base64 dengan format gambar yang berbeda"""
        formats = ['RGB', 'RGBA', 'L']  # Warna, Warna+Alpha, Grayscale
        for format in formats:
            test_image = Image.new(format, (100, 100), color='red')
            base64_string = image_to_base64(test_image)
            self.assertTrue(isinstance(base64_string, str))

    def test_image_to_base64_invalid_image(self):
        """Menguji image_to_base64 dengan gambar yang tidak valid"""
        with self.assertRaises(AttributeError):
            image_to_base64(None)

class TestEmbedData(unittest.TestCase):
    """Menguji kelas EmbedData"""
    
    @patch('rag_code.ColPali.from_pretrained')
    @patch('rag_code.ColPaliProcessor.from_pretrained')
    def setUp(self, mock_processor, mock_model):
        """Menyiapkan lingkungan uji"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_data = EmbedData(device=self.device)
        self.mock_model = mock_model
        self.mock_processor = mock_processor

    def test_init(self):
        """Menguji inisialisasi EmbedData"""
        self.assertEqual(self.embed_data.embed_model_name, "vidore/colpali-v1.2")
        self.assertEqual(self.embed_data.device, self.device)
        self.assertEqual(self.embed_data.batch_size, 1)

    @patch('torch.no_grad')
    def test_get_query_embedding(self, mock_no_grad):
        """Menguji pembuatan embedding query"""
        test_query = "test query"
        # Memalsukan output prosesor dan model
        self.embed_data.processor.process_queries.return_value = torch.ones(1, 10)
        self.embed_data.embed_model.return_value = torch.ones(1, 128)
        
        embedding = self.embed_data.get_query_embedding(test_query)
        
        self.assertTrue(isinstance(embedding, list))
        self.assertEqual(len(embedding), 128)  # Memeriksa apakah embedding memiliki dimensi yang benar

    def test_embed_empty_images_list(self):
        """Menguji embedding dengan daftar gambar kosong"""
        self.embed_data.embed([])
        self.assertEqual(self.embed_data.embeddings, [])

    @patch('torch.cuda.empty_cache')
    def test_memory_cleanup(self, mock_empty_cache):
        """Menguji bahwa memori GPU dibersihkan dengan benar"""
        test_query = "test query"
        self.embed_data.get_query_embedding(test_query)
        mock_empty_cache.assert_called()

    def test_generate_embedding_batch_size(self):
        """Menguji generate_embedding dengan ukuran batch yang berbeda"""
        test_image = Image.new('RGB', (100, 100), color='red')
        self.embed_data.batch_size = 2
        self.embed_data.processor.process_images.return_value = torch.ones(2, 10)
        self.embed_data.embed_model.return_value = torch.ones(2, 128)
        
        embeddings = self.embed_data.generate_embedding([test_image, test_image])
        self.assertEqual(len(embeddings), 2)

class TestQdrantVDB(unittest.TestCase):
    """Menguji kelas QdrantVDB_QB"""
    
    def setUp(self):
        """Menyiapkan lingkungan uji"""
        self.collection_name = "test_collection"
        self.qdrant = QdrantVDB_QB(collection_name=self.collection_name)

    @patch('rag_code.QdrantClient')
    def test_define_client(self, mock_client):
        """Menguji definisi klien"""
        self.qdrant.define_client()
        mock_client.assert_called_once()

    @patch('rag_code.QdrantClient')
    def test_create_collection(self, mock_client):
        """Menguji pembuatan koleksi"""
        self.qdrant.client = Mock()
        self.qdrant.client.collection_exists.return_value = False
        self.qdrant.create_collection()
        self.qdrant.client.create_collection.assert_called_once()

    def test_create_collection_already_exists(self):
        """Menguji pembuatan koleksi yang sudah ada"""
        self.qdrant.client = Mock()
        self.qdrant.client.collection_exists.return_value = True
        self.qdrant.create_collection()
        self.qdrant.client.create_collection.assert_not_called()

    def test_ingest_data_empty(self):
        """Menguji pengambilan data kosong"""
        mock_embeddata = Mock()
        mock_embeddata.embeddings = []
        mock_embeddata.images = []
        self.qdrant.ingest_data(mock_embeddata)
        # Tidak boleh menimbulkan kesalahan

    @patch('torch.cuda.empty_cache')
    def test_ingest_data_memory_cleanup(self, mock_empty_cache):
        """Menguji pembersihan memori selama pengambilan data"""
        mock_embeddata = Mock()
        mock_embeddata.embeddings = [[0.1] * 128]
        mock_embeddata.images = [Image.new('RGB', (100, 100), color='red')]
        self.qdrant.client = Mock()
        self.qdrant.ingest_data(mock_embeddata)
        mock_empty_cache.assert_called()

class TestRetriever(unittest.TestCase):
    """Menguji kelas Retriever"""
    
    def setUp(self):
        """Menyiapkan lingkungan uji"""
        self.mock_vector_db = Mock()
        self.mock_embeddata = Mock()
        self.retriever = Retriever(self.mock_vector_db, self.mock_embeddata)

    def test_search(self):
        """Menguji fungsionalitas pencarian"""
        test_query = "test query"
        test_embedding = [0.1] * 128
        self.mock_embeddata.get_query_embedding.return_value = test_embedding
        
        self.retriever.search(test_query)
        
        self.mock_embeddata.get_query_embedding.assert_called_with(test_query)
        self.mock_vector_db.client.query_points.assert_called_once()

    def test_search_no_results(self):
        """Menguji pencarian tanpa hasil"""
        test_query = "test query"
        self.mock_embeddata.get_query_embedding.return_value = [0.1] * 128
        self.mock_vector_db.client.query_points.return_value = Mock(points=[])
        
        result = self.retriever.search(test_query)
        self.assertEqual(len(result.points), 0)

    def test_search_with_limit(self):
        """Menguji pencarian dengan batas hasil yang berbeda"""
        test_query = "test query"
        self.mock_embeddata.get_query_embedding.return_value = [0.1] * 128
        self.retriever.search(test_query)
        
        # Memverifikasi bahwa limit=4 digunakan dalam pencarian
        self.mock_vector_db.client.query_points.assert_called_with(
            collection_name=self.mock_vector_db.collection_name,
            query=[0.1] * 128,
            limit=4,
            search_params=unittest.mock.ANY
        )

class TestRAG(unittest.TestCase):
    """Menguji kelas RAG"""
    
    @patch('rag_code.VLChatProcessor.from_pretrained')
    @patch('rag_code.AutoModelForCausalLM.from_pretrained')
    def setUp(self, mock_model, mock_processor):
        """Menyiapkan lingkungan uji"""
        self.mock_retriever = Mock()
        self.mock_retriever.embeddata.device = "cpu"
        self.rag = RAG(self.mock_retriever)

    def test_generate_context(self):
        """Menguji pembuatan konteks"""
        test_query = "test query"
        mock_result = Mock()
        mock_result.points = [Mock(payload={"image": "test_image_base64"})]
        self.mock_retriever.search.return_value = mock_result
        
        context = self.rag.generate_context(test_query)
        
        self.assertEqual(context, "test_image_base64")
        self.mock_retriever.search.assert_called_with(test_query)

    def test_generate_context_no_results(self):
        """Menguji pembuatan konteks tanpa hasil pencarian"""
        test_query = "test query"
        mock_result = Mock()
        mock_result.points = []
        self.mock_retriever.search.return_value = mock_result
        
        context = self.rag.generate_context(test_query)
        self.assertEqual(context, "No relevant images found.")

    def test_set_template(self):
        """Menguji pengaturan template"""
        test_template = "Answer: [response]"
        self.rag.set_template(test_template)
        self.assertEqual(self.rag.template, test_template)
        self.assertEqual(self.rag.template_placeholders, ["response"])

    def test_process_template_no_template(self):
        """Menguji process_template ketika tidak ada template yang diatur"""
        test_text = "Test response"
        self.rag.template = None
        result = self.rag.process_template(test_text)
        self.assertEqual(result, test_text)

    def test_process_template_multiple_placeholders(self):
        """Menguji pemrosesan template dengan beberapa placeholder"""
        test_template = "Question: [question] Answer: [answer] Response: [response]"
        test_text = "This is a test response"
        self.rag.set_template(test_template)
        result = self.rag.process_template(test_text)
        expected = "Question: This is a test response Answer: This is a test response Response: This is a test response"
        self.assertEqual(result, expected)

    def test_feedback_logging(self):
        """Menguji fungsionalitas logging feedback"""
        with self.assertLogs(level='INFO') as log:
            self.rag.feedback("test query", "positive")
            self.assertTrue(any("Feedback received for query 'test query': positive" in output for output in log.output))

def main():
    unittest.main()

if __name__ == '__main__':
    main() 