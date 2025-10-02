use std::mem;
use std::sync::atomic::{AtomicU64, Ordering};

///from https://github.com/wjakob/dset, changed to add connected component
///computation
pub struct DisjointSets
{
	m_data: Vec<AtomicU64>,
}

impl DisjointSets
{
	pub fn new(size: u32) -> Self
	{
		Self
		{
			m_data: (0..size).map(|i| AtomicU64::new(i as u64)).collect(),
		}
	}
	
	pub fn find(&self, mut id: u32) -> u32
	{
		while id != self.parent(id)
		{
			let value = self.m_data[id as usize].load(Ordering::SeqCst);
			let new_parent = self.parent(value as u32);
			let new_value = (value & 0xFFFFFFFF00000000) | (new_parent as u64);
			if value != new_value
			{
				/* Try to update parent (may fail, that's ok) */
				#[allow(unused_must_use)]
				self.m_data[id as usize].compare_exchange_weak
				(
					value,
					new_value,
					Ordering::SeqCst,
					Ordering::SeqCst,
				);
			}
			
			id = new_parent;
		}
		
		id
	}
	
	pub fn unite(&self, mut id1: u32, mut id2: u32) -> u32
	{
		loop
		{
			id1 = self.find(id1);
			id2 = self.find(id2);
			
			if id1 == id2 { return id1; }
			
			let mut r1 = self.rank(id1);
			let mut r2 = self.rank(id2);
			
			if r1 > r2 || (r1 == r2 && id1 < id2)
			{
				mem::swap(&mut r1, &mut r2);
				mem::swap(&mut id1, &mut id2);
			}
			
			let mut old_entry = ((r1 as u64) << 32) | (id1 as u64);
			let mut new_entry = ((r1 as u64) << 32) | (id2 as u64);
			
			if self.m_data[id1 as usize].compare_exchange
			(
				old_entry,
				new_entry,
				Ordering::SeqCst,
				Ordering::SeqCst,
			).is_err()
			{
				continue;
			}
			
			if r1 == r2
			{
				old_entry = ((r2 as u64) << 32) | (id2 as u64);
				new_entry = (((r2 as u64) + 1) << 32) | (id2 as u64);
				/* Try to update the rank (may fail, retry if rank = 0) */
				if self.m_data[id2 as usize].compare_exchange
				(
					old_entry,
					new_entry,
					Ordering::SeqCst,
					Ordering::SeqCst,
				).is_err() && r2 == 0
				{
					continue;
				}
			}
			
			break;
		}
		
		id2
	}
	
	fn rank(&self, id: u32) -> u32
	{
		((self.m_data[id as usize].load(Ordering::SeqCst) >> 32) as u32) & 0x7FFFFFFF
	}
	
	fn parent(&self, id: u32) -> u32
	{
		self.m_data[id as usize].load(Ordering::SeqCst) as u32
	}
}
